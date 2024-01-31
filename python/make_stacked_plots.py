"""
Loads the config from `config_make_stacked_plots.yaml` and postprocesses the condor output to make stacked histogram plots.

It does so in two steps,
    1. `make_events_dict()`: which builds a massive pandas dataframe (expensive but only done once)
    2. `make_plots_from_events_dict()`: will make plots of the relevant regions of interest specefied in the config

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def make_events_dict(years, channels, samples_dir, samples, presel, logging_=True):
    """
    Postprocess the parquets by applying preselections, saving an `event_weight` column, and
    a tagger score column in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [Dict]: keys are years, and values are the path of the parquets for that year
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)

    Returns
        a dict() object events_dict[year][channel][samples] that contains big dataframes of procesed events

    """

    if logging_ is False:
        logger = logging.getLogger()
        logger.disabled = True

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            events_dict[year][ch] = {}

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):
                # get a combined label to combine samples of the same process

                if "DYJetsToLL_M-50_HT" in sample:
                    print(f"Skipping sample {sample}")
                    continue

                for key in utils.combine_samples:
                    if key in sample:
                        sample_to_use = utils.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                if "postprocess" in samples_dir[year]:
                    parquet_files = glob.glob(f"{out_files}/{ch}.parquet")
                else:
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")

                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)

                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # drop unnecessary columns
                data = data[data.columns.drop(list(data.filter(regex="weight_mu_")))]
                data = data[data.columns.drop(list(data.filter(regex="weight_ele_")))]
                data = data[data.columns.drop(list(data.filter(regex="L_btag")))]
                data = data[data.columns.drop(list(data.filter(regex="M_btag")))]
                data = data[data.columns.drop(list(data.filter(regex="T_btag")))]
                data = data[data.columns.drop(list(data.filter(regex="veto")))]
                data = data[data.columns.drop(list(data.filter(regex="fj_H_VV_")))]
                data = data[data.columns.drop(list(data.filter(regex="_up")))]
                data = data[data.columns.drop(list(data.filter(regex="_down")))]

                data["abs_met_fj_dphi"] = np.abs(data["met_fj_dphi"])

                # get event_weight
                if sample_to_use != "Data":
                    event_weight = utils.get_xsecweight(pkl_files, year, sample, False, luminosity)
                    logging.info("---> Using already stored event weight")
                    event_weight *= data[f"weight_{ch}"]
                else:
                    event_weight = np.ones_like(data["fj_pt"])
                data["event_weight"] = event_weight

                # use hidNeurons to get the finetuned scores
                data["fj_ParT_score_finetuned"] = utils.get_finetuned_score(data, modelv="v2_nor2")

                # drop hidNeuron columns for memory
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply any pre-selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                logging.info(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
                logging.info(f"tot event weight {data['event_weight'].sum()} \n")

                # fill the big dataframe
                if sample_to_use not in events_dict[year][ch]:
                    events_dict[year][ch][sample_to_use] = data
                else:
                    events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data])

    return events_dict


def make_plots_from_events_dict(events_dict, config, PATH):
    """
    Takes an `events_dict` object that was processed by `make_events_dict` and starts filling histograms.

    Args
        events_dict [dict]: see output of `make_events_dict()`
        config [dictionnary]: contains info such as years_to_plot, vars_to_plot, regions_to_plot, etc.
        PATH [str]: the path at which to store the plots

    """

    hists = {}
    for region, selection in config["regions_to_plot"].items():
        hists[region] = {}
        for var in config["vars_to_plot"]:
            hists[region][var] = hist2.Hist(
                hist2.axis.StrCategory([], name="samples", growth=True),
                utils.axis_dict[var],
            )

            for sample in config["samples_to_plot"]:
                for year in config["years_to_plot"]:
                    for ch in config["channels_to_plot"]:
                        df = events_dict[year][ch][sample]

                        df = df.query(selection)

                        hists[region][var].fill(
                            samples=sample,
                            var=df[var],
                            weight=df["event_weight"],
                        )

        for var in config["vars_to_plot"]:
            fix_neg_yields(hists[region][var])

        # for each region save the plots in a different directory
        PATH_region = PATH + "/" + region + "/"
        if not os.path.exists(PATH_region):
            os.makedirs(PATH_region)

        print(PATH_region)
        os.system(f"cp config_make_stacked_plots.yaml {PATH_region}")

        utils.plot_hists(
            hists[region],
            config["years_to_plot"],
            config["channels_to_plot"],
            config["vars_to_plot"],
            config["add_data"],
            config["logy"],
            config["add_soverb"],
            config["only_sig"],
            config["mult"],
            outpath=PATH_region,
        )


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for sample in h.axes["samples"]:
        neg_bins = np.where(h[{"samples": sample}].values() < 0)[0]

        if len(neg_bins) > 0:
            print(f"{sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

            sample_index = np.argmax(np.array(h.axes["samples"]) == sample)

            for neg_bin in neg_bins:
                h.view(flow=True)[sample_index, neg_bin + 1] = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    PATH = args.outpath + f"stacked_hists_{args.tag}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    os.system(f"cp config_make_stacked_plots.yaml {PATH}/")

    # load config from yaml
    with open("config_make_stacked_plots.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.make_events_dict:
        events_dict = make_events_dict(
            years,
            channels,
            config["samples_dir"],
            config["samples"],
            config["presel"],
        )
        with open(f"{PATH}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{PATH}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.info("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    if args.plot_hists:
        make_plots_from_events_dict(events_dict, config, PATH)


if __name__ == "__main__":
    # e.g. to build the events dict
    # python make_stacked_plots.py --years 2018,2017,2016,2016APV --channels ele,mu --make_events_dict --tag v1

    # e.g. to make the plots (edit the `config_make_stacked_plots.yaml` first)
    # python make_stacked_plots.py --make_plots --tag v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--outpath", dest="outpath", default="hists/", help="path of the output", type=str)
    parser.add_argument("--tag", dest="tag", default="test/", help="path of the output", type=str)
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="make events dictionary", action="store_true")
    parser.add_argument("--make_plots", dest="plot_hists", help="plot histograms", action="store_true")

    args = parser.parse_args()

    main(args)
