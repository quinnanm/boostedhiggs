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

from sklearn.metrics import auc, roc_curve
import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})


def construct_score_arrays(ch, idir, odir, weights, presel, samples):
    """
    Makes nd arrays containing the tagger scores for each of the three classes (Higgs, QCD, Top).

     Args:
         ch: string that represents the signal channel to look at... choices are ['ele', 'mu'].
         idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet).
         odir: output directory to hold the hist object.
         weights: weights to be applied to MC.
         presel: pre-selection string.
         samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json).
    """

    # define the scores
    scores = [
        "hww_score (H)",
        "H/(1-H)",
        "H/(H+QCD)",
        "H/(H+Top)",
    ]  # , "qcd_score (QCD)", "top_score (Top)"]

    tagger_scores = {}
    for score in scores:
        tagger_scores[score] = []
    label_array = []

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
                        HIGGS += df_all_softmax[:, i]

                # filling the scores nd arrays
                for score in scores:
                    if score == "hww_score (H)":
                        tagger_scores[score] += np.ndarray.tolist(HIGGS)
                    elif score == "H/(1-H)":
                        tagger_scores[score] += np.ndarray.tolist(HIGGS / (1 - HIGGS))
                    elif score == "H/(H+QCD)":
                        tagger_scores[score] += np.ndarray.tolist(HIGGS / (HIGGS + QCD))
                    elif score == "H/(H+Top)":
                        tagger_scores[score] += np.ndarray.tolist(HIGGS / (HIGGS + TOP))

                if "HToWW" in sample_to_use:
                    label = 1
                else:
                    label = 0
                label_array += [label] * len(
                    np.ndarray.tolist(HIGGS)
                )  # just the length of the events

    # store the scores
    with open(f"{odir}/{ch}_tagger_scores.pkl", "wb") as f:
        pkl.dump(tagger_scores, f)
    # store the labels to make ROC curves
    with open(f"{odir}/{ch}_tagger_labels.pkl", "wb") as f:
        pkl.dump(label_array, f)


def make_roc_curves(odir, ch):
    with open(f"{odir}/{ch}_tagger_labels.pkl", "rb") as f:
        labels = pkl.load(f)

    with open(f"{odir}/{ch}_tagger_scores.pkl", "rb") as f:
        scores = pkl.load(f)

    def make_roc(labels, scores, ch):
        fig, ax = plt.subplots()
        for key in scores.keys():  # keys here are the score classes (H, H/1-H, ...)
            # get nan scores indices
            nan_indices = np.argwhere(np.isnan(scores[key]))

            label = np.delete(np.array(labels), nan_indices)
            score = np.delete(np.array(scores[key]), nan_indices)

            fpr, tpr, _ = roc_curve(label, score)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                tpr,
                fpr,
                lw=2,
                label=f"{key} - AUC = {round(auc(fpr, tpr)*100,2)}%",
            )
        plt.xlim([0.0, 1.0])
        plt.ylabel("False Positive Rate")
        plt.xlabel("True Positive Rate")
        plt.yscale("log")
        plt.legend(title=f"{ch} channel", loc="lower right")
        plt.savefig(f"{odir}/{ch}_roc.png")
        print(f"saved plot under {odir}/{ch}_roc.png")

    make_roc(labels, scores, ch)


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
        print(f"Making TAGGER nd score arrays for {ch}...")
        print("Weights: ", weights[ch])
        print("Pre-selection: ", presel[ch])
        construct_score_arrays(ch, args.idir, odir, weights[ch], presel[ch], samples)

        print(f"Plotting ROC curves for {ch}...")
        make_roc_curves(odir, ch)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_roc_curves.py --year 2017 --odir Dec15tagger --channels ele --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov16_inference

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
