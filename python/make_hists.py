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
    axis_dict,
    data_by_ch,
    data_by_ch_2018,
    get_cutflow,
    get_cutflow_axis,
    get_sample_to_use,
    get_xsecweight,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})


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
        # data label and lumi
        data_label = data_by_ch[ch]
        if yr == "2018":
            data_label = data_by_ch_2018[ch]
        f = open("../fileset/luminosity.json")
        luminosity = json.load(f)[ch][yr]
        f.close()
        print(
            f"Processing samples from year {yr} with luminosity {luminosity} for channel {ch}"
        )

        for sample in samples[yr][ch]:
            # print(f"Sample {sample}")

            # check if the sample was processed
            pkl_dir = f"{idir}_{yr}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)
            if not pkl_files:  # skip samples which were not processed
                print(
                    "- No processed files found...",
                    pkl_dir,
                    "skipping sample...",
                    sample,
                )
                continue

            # get list of parquet files that have been post processed
            parquet_files = glob.glob(f"{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet")

            # define an is_data boolean
            is_data = False
            if data_label in sample:
                is_data = True

            # get combined sample
            sample_to_use = get_sample_to_use(sample, yr, is_data)

            # xsec weight
            xsec_weight = get_xsecweight(pkl_files, yr, sample, is_data, luminosity)

            # get cutflow
            if sample_to_use not in values.keys():
                values[sample_to_use] = dict.fromkeys(cut_keys, 0)

            cutflow = get_cutflow(cut_keys, pkl_files, yr, sample, xsec_weight, ch)
            # print("cutflow ", cutflow)
            
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
                        print(
                            "Not able to read data: ",
                            parquet_file,
                            " should remove events from scaling/lumi",
                        )
                    else:
                        print("Not able to read data from ", parquet_file)
                    continue

                # print parquet content
                # if i==0:
                #    print(sample, data.columns)

                if len(data) == 0:
                    print(
                        f"WARNING: Parquet file empty {yr} {ch} {sample} {parquet_file}"
                    )
                    continue

                # modify dataframe with string queries
                if presel is not None:
                    for sel_key, sel_str in presel.items():
                        df = data.query(sel_str)
                        weight_ones = np.ones_like(df["fj_pt"])
                        #print(sel_key,sel_str,np.sum(weight_ones),np.sum(weight_ones * xsec_weight))
                        values[sample_to_use][sel_key] += np.sum(weight_ones * xsec_weight)

                # get event weight
                if not is_data:
                    event_weight = xsec_weight
                    for w in weights:
                        try:
                            event_weight *= data[w]
                        except ValueError:
                            print_warning = True
                            if w == "weight_vjets_nominal" or (
                                w == "weight_L1Prefiring" and yr == "2018"
                            ):
                                print_warning = False
                            if print_warning:
                                print(f"No {w} variable in parquet for sample {sample}")
                else:
                    event_weight = np.ones_like(data["fj_pt"])

                for var in vars_to_plot:
                    if var == "cutflow":
                        continue

                    if var not in data.keys():
                        if "gen" in var:
                            continue
                        print(f"Var {var} not in parquet keys")
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
                hists["cutflow"].fill(
                    samples=sample_to_use, var=cut_index, weight=numevents
                )

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
    global axis_dict
    extra_cut_keys = []
    if "selection" in variables.keys():
        extra_cut_keys = list(variables["selection"].keys())
    axis_dict["cutflow"] = get_cutflow_axis(cut_keys + extra_cut_keys)

    axis_sample = hist2.axis.StrCategory([], name="samples", growth=True)

    # extract variables to plot
    vars_to_plot = []
    for key, value in variables["vars"].items():
        if value == 1:
            vars_to_plot.append(key)
    print(f"Variables to include {vars_to_plot}")

    # define channels
    if args.channels == "all":
        channels = ["ele", "mu"]
        hists = {}
        values = {}
        for var in vars_to_plot:
            hists[var] = hist2.Hist(
                axis_sample,
                axis_dict[var],
            )
    else:
        channels = args.channels.split(",")
        hists = None
        values = None

    # extract variables and weights from yaml file
    for ch in channels:
        if ch not in variables.keys():
            raise Exception(f"Channel {ch} not included in yaml file")

        weights = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights.append(key)

        presel = {}
        if "selection" in variables.keys():
            cum_presel_str = "(1==1)"
            for presel_key, presel_str in variables["selection"][ch].items():
                cum_presel_str += "& " + presel_str
                presel[presel_key] = cum_presel_str

        if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
            print("Histograms already exist - remaking them")

        print(f"Making histograms for {ch}...")
        print("Weights: ", weights)
        print("Pre-selection: ", presel.keys())

        if hists is None:
            hists_per_ch = {}
            for var in vars_to_plot:
                hists_per_ch[var] = hist2.Hist(
                    axis_sample,
                    axis_dict[var],
                )
        else:
            print(f"Filling the same histogram to combine all channels")
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
        # print(values_per_ch)

        if args.channels != "all":
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

        print(f"Finished histograms for {ch} \n")

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
        default="trigger,leptonKin,fatjetKin,ht,oneLepton,notaus,leptonInJet",
        help="cut keys for cutflow (split by commas)",
    )

    args = parser.parse_args()

    main(args)
