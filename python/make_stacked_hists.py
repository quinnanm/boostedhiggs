"""
Loads the config from `config_make_stacked_hists.yaml`, and postprocesses
the condor output to make stacked histograms.

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


def make_events_dict(years, channels, samples_dir, samples, presel, weights):
    """
    Postprocess the parquets by applying preselections, saving an `event_weight` column, and
    a tagger score column in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets (note: the year will be appended to the string)
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)
        weights [dict]: weights to include in the event_weight per ch (e.g. `weights = {"mu": {"weight_genweight": 1}})

    Returns
        a dict() object events_dict[year][channel][samples] that contains big dataframes of procesed events

    """

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            events_dict[year][ch] = {}

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            condor_dir = os.listdir(samples_dir + year)

            for sample in condor_dir:
                if sample == "DYJetsToLL_M-10to50":
                    continue  # because tagger didnt run for it for one of the years

                # get a combined label to combine samples of the same process
                for key in utils.combine_samples:
                    if key in sample:
                        sample_to_use = utils.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir + year}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)

                except pyarrow.lib.ArrowInvalid:
                    # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # # replace the weight_pileup of the strange events with the mean weight_pileup of all the other events
                # # TODO: draw distribution of number of primary vertices before and after applying this weight
                # if sample_to_use != "Data":
                #     strange_events = data["weight_pileup"] > 4
                #     if len(strange_events) > 0:
                #         data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

                # get event_weight
                if sample_to_use != "Data":
                    event_weight = utils.get_xsecweight(pkl_files, year, sample, False, luminosity)

                    if (
                        ("Apr12_presel" in samples_dir)
                        or ("Jul12QCD" in samples_dir)
                        or ("Jul15_region_wjets" in samples_dir)
                    ):
                        logging.info("---> Accumulating event weights")
                        for w in weights[ch]:
                            if w not in data.keys():
                                logging.info(f"{w} weight is not stored in parquet")
                                continue
                            if weights[ch][w] == 1:
                                logging.info(f"Applying {w} weight")
                                event_weight *= data[w]

                        logging.info("---> Done with accumulating event weights")
                    else:
                        logging.info("---> Using already stored event weight")
                        event_weight *= data[f"weight_{ch}"]

                else:
                    event_weight = np.ones_like(data["fj_pt"])

                data["event_weight"] = event_weight

                # add tagger score
                if "Apr12_presel" in samples_dir:
                    data["inclusive_score"] = utils.disc_score(data, utils.new_sig, utils.inclusive_bkg)
                else:
                    data["inclusive_score"] = data["fj_ParT_all_score"]

                # apply selection
                logging.info("---> Applying preselection")

                for selection in presel[ch]:
                    logging.info(f"applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                logging.info("---> Done with preselection")

                logging.info(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
                logging.info(f"tot event weight {data['event_weight'].sum()} \n")

                # fill the big dataframe
                if sample_to_use not in events_dict[year][ch]:
                    events_dict[year][ch][sample_to_use] = data
                else:
                    events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data])

    return events_dict


def make_hists_from_events_dict(events_dict, samples_to_plot, vars_to_plot):
    """
    Takes an `events_dict` object that was processed by `make_events_dict` and starts filling histograms.

    Args
        events_dict [dict]: see output of `make_events_dict()`
        samples_to_plot [list]: which samples to use when plotting
        vars_to_plot [list]: which variables to plot

    Returns
        hist.Hist object

    """

    hists = {}
    for var in vars_to_plot:
        hists[var] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            utils.axis_dict[var],
        )

        for sample in samples_to_plot:
            for year in events_dict:
                for ch in events_dict[year]:
                    df = events_dict[year][ch][sample]

                    hists[var].fill(samples=sample, var=df[var], weight=df["event_weight"])

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp config_make_stacked_hists.yaml {args.outpath}/")

    # load config from yaml
    with open("config_make_stacked_hists.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.make_events_dict:
        events_dict = make_events_dict(
            years,
            channels,
            args.samples_dir,
            config["samples"],
            config["presel"],
            config["weights"],
        )
        with open(f"{args.outpath}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{args.outpath}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.info("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    if args.plot_hists:
        hists = make_hists_from_events_dict(events_dict, config["samples_to_plot"], config["vars_to_plot"])

        utils.plot_hists(
            years,
            channels,
            hists,
            config["vars_to_plot"],
            config["add_data"],
            config["logy"],
            config["add_soverb"],
            config["only_sig"],
            config["mult"],
            outpath=f"{args.outpath}/stacked_hists/",
        )


if __name__ == "__main__":
    # e.g.
    # python make_stacked_hists.py --years 2017 --channels ele,mu --make_events_dict --plot_hists

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="ele", help="channels separated by commas")
    parser.add_argument(
        "--samples_dir", dest="samples_dir", default="../eos/Apr12_presel_", help="path to parquets", type=str
    )
    parser.add_argument(
        "--outpath", dest="outpath", default="/Users/fmokhtar/Desktop/hww/test", help="path of the output", type=str
    )
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="Make events dictionary", action="store_true")
    parser.add_argument("--plot_hists", dest="plot_hists", help="Plot histograms", action="store_true")

    args = parser.parse_args()

    main(args)
