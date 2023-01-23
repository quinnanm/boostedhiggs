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
import scipy
import json
import os, glob, sys
import argparse

import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})


def make_hists(ch, idir, odir, weights, presel, samples):
    """
    Makes 1D histograms of the tagger scores to be plotted as stacked over the different samples.

    Args:
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu'].
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet).
        odir: output directory to hold the hist object.
        vars_to_plot: the set of variables to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json).
        presel: pre-selection string.
        weights: weights to be applied to MC.
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json).
    """

    # define histograms
    hists = {}
    sample_axis = hist2.axis.StrCategory([], name="samples", growth=True)
    plot_vars = ["qcd_score", "top_score", "hww_score"]
    for var in plot_vars:
        hists[var] = hist2.Hist(
            sample_axis,
            hist2.axis.Regular(35, 0, 1, name="var", label=var, overflow=True),
        )

    hists["fj_pt"] = hist2.Hist(
        sample_axis,
        axis_dict["fj_pt"],
    )

    for score in scores:
        hists[score] = hist2.Hist(
            sample_axis,
            hist2.axis.Regular(35, 0, 1, name="score", label=score, overflow=True),
        )

    labels = []
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
            if "ttHToNonbb_M125" in sample:  # skip ttH cz labels are not stored
                continue
            if data_label in sample:  # skip data samples
                continue
            is_data = False

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

            # get combined sample
            sample_to_use = get_sample_to_use(sample, yr)

            # get cutflow
            xsec_weight = get_xsecweight(pkl_files, yr, sample, is_data, luminosity)

            for parquet_file in parquet_files:
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    print("Not able to read data from ", parquet_file)
                    continue

                # retrieve tagger labels from one parquet
                if len(labels) == 0:
                    for key in data.keys():
                        if "label" in key:
                            labels.append(key)
                            print(key)

                if len(data) == 0:
                    print(
                        f"WARNING: Parquet file empty {yr} {ch} {sample} {parquet_file}"
                    )
                    continue

                # modify dataframe with pre-selection query
                if presel is not None:
                    data = data.query(presel)

                event_weight = xsec_weight
                weight_ones = np.ones_like(data["weight_genweight"])
                for w in weights:
                    try:
                        event_weight *= data[w]
                    except:
                        if w != "weight_vjets_nominal":
                            print(f"No {w} variable in parquet for sample {sample}")

                # save fj_pt to order the stack plots
                hists["fj_pt"].fill(
                    samples=sample_to_use,
                    var=data["fj_pt"],
                    weight=event_weight,
                )

                data = data[labels]  # keep only the labels

                # apply softmax per row
                df_all_softmax = scipy.special.softmax(data[labels].values, axis=1)

                # combine the labels under QCD, TOP or HWW
                QCD = 0
                TOP = 0
                HIGGS = 0

                for i, label in enumerate(labels):  # labels are the tagger classes
                    if "QCD" in label:
                        QCD += df_all_softmax[:, i]
                    elif "Top" in label:
                        TOP += df_all_softmax[:, i]
                    else:
                        scores["hww_score"] += df_all_softmax[:, i]

                # filling histograms
                for var in plot_vars:
                    hists[var].fill(
                        samples=sample_to_use,
                        var=scores[var],
                        weight=event_weight,
                    )

                # filling histograms
                for score in scores:
                    if score == "qcd_score (QCD)":
                        X = QCD
                    elif score == "top_score (Top)":
                        X = TOP
                    elif score == "hww_score (H)":
                        X = HIGGS
                    elif score == "H/(1-H)":
                        X = HIGGS / (1 - HIGGS)
                    elif score == "H/(H+QCD)":
                        X = HIGGS / (HIGGS + QCD)
                    elif score == "H/(H+Top)":
                        X = HIGGS / (HIGGS + TOP)

                    hists[score].fill(
                        samples=sample_to_use,
                        score=X,
                        weight=event_weight,
                    )
    # store the hists variable
    with open(f"{odir}/{ch}_hists_tagger.pkl", "wb") as f:
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

    # list of weights to apply to MC
    weights = {}
    # pre-selection string
    presel = {}
    for ch in channels:

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        presel_str = None
        if type(variables[ch]["pre-sel"]) is list:
            presel_str = variables[ch]["pre-sel"][0]
            for i, sel in enumerate(variables[ch]["pre-sel"]):
                if i == 0:
                    continue
                presel_str += f"& {sel}"
        presel[ch] = presel_str

    os.system(f"cp {args.vars} {odir}/")

    for ch in channels:
        if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
            print("Histograms already exist - remaking them")
        print(f"Making TAGGER histograms for {ch}...")
        print("Weights: ", weights[ch])
        print("Pre-selection: ", presel[ch])
        make_hists(ch, args.idir, odir, weights[ch], presel[ch], samples)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_hists_tagger.py --year 2017 --odir Nov23tagger --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov16_inference

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
        default="plot_configs/vars.yaml",
        help="path to json with variables to be plotted",
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
        "--add_score", dest="add_score", action="store_true", help="Add inference score"
    )

    args = parser.parse_args()

    main(args)
