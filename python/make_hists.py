import argparse
import glob
import json
import os
import pickle as pkl

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pyarrow.parquet as pq
import yaml
from utils import (
    data_ref,
    axis_dict,
    get_cutflow,
    get_cutflow_axis,
    get_sample_to_use,
    get_xsecweight,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

import logging


def make_hists(ch, idir, odir, vars_to_plot, weights, presel, samples, cut_keys, hists):
    """
    Makes 1D histograms of the "vars_to_plot" to be plotted as stacked over the different samples.

    Args:
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu'].
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet).
        odir: output directory to hold the hist object.
        vars_to_plot: the set of variables to plot 1D-histograms of.
        presel: pre-selection dictionary.
        weights: weights to be applied to MC.
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_all.json).
        cut_keys: cut keys
        hists: histogram dictionary to fill
    """

    # dictionary to store cutflow
    values = {}

    # loop over the years
    for yr in samples.keys():

        # get luminosity per year
        f = open("../fileset/luminosity.json")
        luminosity = json.load(f)[ch][yr]
        f.close()
        logger.info(f"Processing samples from year {yr} with luminosity {luminosity} for channel {ch}")

        # loop over samples
        for sample in samples[yr][ch]:
            logger.debug(f"Sample {sample}")

            # check if the sample was processed
            pkl_dir = f"{idir}_{yr}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)
            if not pkl_files:  # skip samples which were not processed
                logger.warning(f"- No processed files found... {pkl_dir} skipping sample... {sample}")
                continue

            # get list of parquet files that have been post processed
            parquet_files = glob.glob(f"{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet")

            # define an is_data boolean
            is_data = False
            for data_label in data_ref:
                if data_label in sample:
                    is_data = True

            # get name of sample to use (allows to merge samples)
            sample_to_use = get_sample_to_use(sample, yr, is_data)

            # xsec weight
            xsec_weight = get_xsecweight(pkl_files, yr, sample, is_data, luminosity)

            # get cutflow
            cutflow = get_cutflow(cut_keys, pkl_files, yr, sample, xsec_weight, ch)
            # print("cutflow ", cutflow)

            if sample_to_use not in values.keys():
                values[sample_to_use] = dict.fromkeys(cut_keys, 0)
            for key, val in cutflow.items():
                values[sample_to_use][key] += val

            if presel is not None:
                for key in presel.keys():
                    values[sample_to_use][key] = 0

            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except ValueError:
                    if is_data:
                        logger.warning(f"Not able to read data: {parquet_file} should remove events from scaling/lumi")
                    else:
                        logger.warning(f"Not able to read data from {parquet_file}")
                    continue

                # print parquet content
                # if i==0:
                #    print(sample, data.columns)

                if len(data) == 0:
                    logger.warning(f"Parquet file empty {yr} {ch} {sample} {parquet_file}")
                    continue

                # modify dataframe with string queries
                if presel is not None:
                    for sel_key, sel_str in presel.items():
                        df = data.query(sel_str)
                        if not is_data:
                            try:
                                genweight = df["weight_genweight"]
                            except ValueError:
                                logger.warning("weight weight_genweight not found in parquet")
                                continue
                            values[sample_to_use][sel_key] += np.sum(genweight * xsec_weight)
                        else:
                            weight_ones = np.ones_like(df["fj_pt"])
                            values[sample_to_use][sel_key] += np.sum(weight_ones * xsec_weight)
                        # print(sel_key,sel_str,np.sum(weight_ones),np.sum(weight_ones * xsec_weight))

                # get event weight
                if not is_data:
                    event_weight = xsec_weight
                    for w in weights:
                        try:
                            weight = data[w]
                        except ValueError:
                            print_warning = True
                            if w == "weight_vjets_nominal" or (w == "weight_L1Prefiring" and yr == "2018"):
                                print_warning = False
                            if print_warning:
                                logger.warning(f"No {w} variable in parquet for sample {sample}")
                        event_weight *= weight
                else:
                    event_weight = np.ones_like(data["fj_pt"])

                for var in vars_to_plot:
                    if var == "cutflow":
                        continue

                    if var not in data.keys():
                        if "gen" in var:
                            continue
                        logger.warning(f"Var {var} not in parquet keys")
                        continue

                    # filling histograms
                    hists[var].fill(
                        samples=sample_to_use,
                        var=data[var],
                        weight=event_weight,
                    )

            # fill cutflow histogram once we have all the values
            for key, numevents in values[sample_to_use].items():
                cut_index = list(values[sample_to_use].keys()).index(key)
                # print("fill histogram ", cut_index, numevents)
                hists["cutflow"].fill(samples=sample_to_use, var=cut_index, weight=numevents)

    return hists, values


def main(args):
    # append '/year' to the output directory
    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        os.system(f"mkdir -p {odir}")

    # get year
    years = ["2016", "2016APV", "2017", "2018"] if args.year == "Run2" else [args.year]

    # get json file with list of samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in json_samples[year]:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    # cut keys
    cut_keys = args.cut_keys.split(",")

    # load yaml config file
    with open(args.vars) as f:
        variables = yaml.safe_load(f)
    os.system(f"cp {args.vars} {odir}/")

    # extract extra cut keys from yaml file
    axes = axis_dict
    extra_cut_keys = []
    if "selection" in variables.keys():
        extra_cut_keys = list(variables["selection"].keys())
    axes["cutflow"] = get_cutflow_axis(cut_keys + extra_cut_keys)

    axis_sample = hist2.axis.StrCategory([], name="samples", growth=True)

    # extract variables to plot
    vars_to_plot = []
    for key, value in variables["vars"].items():
        if value == 1:
            vars_to_plot.append(key)
    logger.info(f"Variables to include {vars_to_plot}")

    # define channels
    if args.channels == "all":
        channels = ["ele", "mu"]
        hists = {}
        values = {}
        for var in vars_to_plot:
            hists[var] = hist2.Hist(
                axis_sample,
                axes[var],
            )
    else:
        channels = args.channels.split(",")
        hists = None
        values = None

    # extract variables and weights from yaml file
    for ch in channels:
        if ch not in variables.keys():
            raise Exception(f"Channel {ch} not included in yaml file")
        logger.info(f"Making histograms for {ch}...")

        weights = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights.append(key)
        logger.info("Weights ", weights)

        presel = {}
        if "selection" in variables.keys():
            cum_presel_str = "(1==1)"
            for presel_key, presel_str in variables["selection"][ch].items():
                cum_presel_str += "& " + presel_str
                presel[presel_key] = cum_presel_str
            logger.info("Pre-selection: %s" % presel.keys())

        if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
            logger.warning("Histograms already exist - remaking them")

        if hists is None:
            hists_per_ch = {}
            for var in vars_to_plot:
                hists_per_ch[var] = hist2.Hist(
                    axis_sample,
                    axes[var],
                )
        else:
            logging.info("Filling the same histogram to combine all channels")
            hists_per_ch = hists

        hists_per_ch, values_per_ch = make_hists(
            ch,
            args.idir,
            odir,
            vars_to_plot,
            weights,
            presel,
            samples,
            cut_keys,
            hists_per_ch,
        )
        print(values_per_ch)
        print(args.channels)

        if args.channels != "all":
            logging.info(f"Saving histograms to {odir}/{ch}_hists.pkl")
            with open(f"{odir}/{ch}_hists.pkl", "wb") as f:
                pkl.dump(hists_per_ch, f)
        else:
            hists = hists_per_ch
            for sample in values_per_ch:
                if sample not in values.keys():
                    values[sample] = {}
                for cutkey, val in values_per_ch[sample].items():
                    if cutkey in values[sample].keys():
                        values[sample][cutkey] += val
                    else:
                        values[sample][cutkey] = val

    if args.channels == "all":
        print(values)
        print(hists["cutflow"])
        with open(f"{odir}/{args.channels}_hists.pkl", "wb") as f:
            pkl.dump(hists, f)


if __name__ == "__main__":

    # e.g. run locally as:
    # # noqa: python make_hists.py --year 2017 --odir Jan23 --channels ele,mu --idir ../Jan20 --vars plot_configs/cutflow.yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        dest="year",
        required=True,
        choices=["2016", "2016APV", "2017", "2018", "Run2"],
        help="year",
    )
    parser.add_argument(
        "--vars",
        dest="vars",
        required=True,
        help="path to yaml with variables to be plotted",
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        default="plot_configs/samples_all.json",
        help="path to json with samples to be plotted",
    )
    parser.add_argument(
        "--channels",
        dest="channels",
        default="ele,mu",
        help="channels for which to plot this variable",
    )
    parser.add_argument(
        "--odir",
        dest="odir",
        default="hists",
        help="tag for output directory... will append '_{year}' to it",
    )
    parser.add_argument(
        "--idir",
        dest="idir",
        default="../results/",
        help="input directory with results - without _{year}",
    )
    parser.add_argument(
        "--cut-keys",
        dest="cut_keys",
        # default="trigger,leptonKin,fatjetKin,ht,oneLepton,notaus,leptonInJet",
        default="trigger",
        help="cut keys for cutflow (split by commas)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=20)
    logger = logging.getLogger("make-hists")

    main(args)
