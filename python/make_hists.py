from utils import (
    axis_dict,
    color_by_sample,
    signal_by_ch,
    data_by_ch,
    data_by_ch_2018,
    label_by_ch,
)
from utils import (
    simplified_labels,
    get_cutflow,
    get_xsecweight,
    get_sample_to_use,
    get_cutflow_axis,
)

import yaml
import pickle as pkl
import pyarrow.parquet as pq
import numpy as np
import json
import os, glob, sys
import argparse

import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})


def make_hists(ch, idir, odir, vars_to_plot, weights, presel, samples, cut_keys):
    """
    Makes 1D histograms of the "vars_to_plot" to be plotted as stacked over the different samples.

    Args:
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu'].
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet).
        odir: output directory to hold the hist object.
        vars_to_plot: the set of variables to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json).
        presel: pre-selection dictionary.
        weights: weights to be applied to MC.
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json).
        cut_keys: cut keys
    """

    # define histograms
    hists = {}
    sample_axis = hist2.axis.StrCategory([], name="samples", growth=True)
    plot_vars = vars_to_plot
    plot_vars.append("cutflow")
    for var in plot_vars:
        hists[var] = hist2.Hist(
            sample_axis,
            axis_dict[var],
        )

    # cutflow dictionary
    cut_values = {}

    # pt cuts for variables
    pt_iso = {"ele": 120, "mu": 55}

    # loop over the samples
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
            print(f"Sample {sample}")

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
            sample_to_use = get_sample_to_use(sample, yr)

            # get cutflow
            xsec_weight = get_xsecweight(pkl_files, yr, sample, is_data, luminosity)
            if sample_to_use not in cut_values.keys():
                cut_values[sample_to_use] = dict.fromkeys(cut_keys, 0)
            cutflow = get_cutflow(cut_keys, pkl_files, yr, sample, xsec_weight, ch)
            for key, val in cutflow.items():
                cut_values[sample_to_use][key] += val

            if presel is not None:
                for key in presel.keys():
                    cut_values[sample_to_use][key] = 0

            sample_yields = {}
            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
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
                        data = data.query(sel_str)
                        if not is_data:
                            event_weight = xsec_weight
                            for w in weights:
                                try:
                                    event_weight *= data[w]
                                except:
                                    pass
                            cut_values[sample_to_use][sel_key] += np.sum(event_weight)
                        else:
                            weight_ones = np.ones_like(data["fj_pt"])
                            cut_values[sample_to_use][sel_key] += np.sum(
                                weight_ones * xsec_weight
                            )

                # get event weight
                if not is_data:
                    event_weight = xsec_weight
                    for w in weights:
                        try:
                            event_weight *= data[w]
                        except:
                            print_warning = True
                            if w == "weight_vjets_nominal" or (
                                w == "weight_L1Prefiring" and yr == "2018"
                            ):
                                print_warning = False
                            if print_warning:
                                print(f"No {w} variable in parquet for sample {sample}")
                else:
                    event_weight = np.ones_like(data["fj_pt"])

                for var in plot_vars:
                    if var == "cutflow":
                        continue

                    if var not in data.keys():
                        if "gen" in var:
                            continue
                        print(f"Var {var} not in parquet keys")
                        continue

                    # filling histograms
                    hists[var].fill(
                        samples=sample_to_use, var=data[var], weight=event_weight
                    )

            # fill cutflow histogram once we have all the values
            for key, numevents in cut_values[sample_to_use].items():
                cut_index = list(cut_values[sample_to_use].keys()).index(key)
                hists["cutflow"].fill(
                    samples=sample_to_use, var=cut_index, weight=numevents
                )

        # print(cut_values)

        # save cutflow values
        with open(f"{odir}/cut_values_{ch}.pkl", "wb") as f:
            pkl.dump(cut_values, f)

    samples = [
        hists["cutflow"].axes[0].value(i)
        for i in range(len(hists["cutflow"].axes[0].edges))
    ]
    print(samples)

    # store the hists variable
    with open(f"{odir}/{ch}_hists.pkl", "wb") as f:
        pkl.dump(hists, f)


def main(args):
    # append '/year' to the output directory
    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        os.system(f"mkdir -p {odir}")

    channels = args.channels.split(",")

    # get year
    years = ["2016", "2016APV", "2017", "2018"] if args.year == "Run2" else [args.year]

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    # load yaml config file
    with open(args.vars) as f:
        variables = yaml.safe_load(f)

    print(variables)

    # variables to plot
    vars_to_plot = {}
    # list of weights to apply to MC
    weights = {}
    # pre-selection dictionary
    presel = {}
    for ch in variables.keys():
        vars_to_plot[ch] = []
        for key, value in variables[ch]["vars"].items():
            if value == 1:
                vars_to_plot[ch].append(key)

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        if "selection" in variables[ch]:
            presel[ch] = variables[ch]["selection"]
        else:
            presel[ch] = None

    cut_keys = args.cut_keys.split(",")
    global axis_dict

    os.system(f"cp {args.vars} {odir}/")

    for ch in channels:
        if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
            print("Histograms already exist - remaking them")

        # get cut keys for cutflow
        extra_cut_keys = []
        if presel[ch] is not None:
            extra_cut_keys = list(presel[ch].keys())
        axis_dict["cutflow"] = get_cutflow_axis(cut_keys + extra_cut_keys)

        print(f"Making histograms for {ch}...")
        print("Weights: ", weights[ch])
        print("Pre-selection: ", presel[ch].keys())
        make_hists(
            ch,
            args.idir,
            odir,
            vars_to_plot[ch],
            weights[ch],
            presel[ch],
            samples,
            cut_keys,
        )


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_hists.py --year 2017 --odir Jan23 --channels ele,mu --idir ../Jan20 --vars plot_configs/cutflow.yaml

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
        default="plot_configs/samples_pfnano.json",
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
