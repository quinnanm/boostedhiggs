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


def compute_soverb(year, hists, ch, range_min=0, range_max=150, remove_ttH=True):
    """
    Computes soverb over range [range_min, range_max] of (jet-lep).mass.
    Assumes lep_fj_m is an axis of the histogram.
    Args:
        year: string that represents the year the processed samples are from.
        hists: the loaded hists.pkl file which contains a Hist() object that has "lep_fj_m" as one of the axes.
        ch: string that represents the signal channel to look at... choices are ["ele", "mu"].
    """

    # data label
    data_label = data_by_ch[ch]
    if year == "2018":
        data_label = data_by_ch_2018[ch]
    elif year == "Run2":
        data_label = "Data"

    # get histograms
    try:
        h = hists["lep_fj_m"]
    except:
        h = hists

    # get samples existing in histogram
    samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
    signal_labels = [label for label in samples if label in signal_by_ch[ch]]
    bkg_labels = [
        label
        for label in samples
        if (label and label != data_label and label not in signal_labels)
    ]

    if remove_ttH:
        signal_labels.remove("ttHToNonbb_M125")

    # data
    data = None
    if data_label in h.axes[0]:
        data = h[{"samples": data_label}]

    # signal
    signal = [h[{"samples": label}] for label in signal_labels]
    # sum all of the signal
    if len(signal) > 0:
        tot_signal = None
        for i, sig in enumerate(signal):
            if tot_signal == None:
                tot_signal = signal[i].copy()
            else:
                tot_signal = tot_signal + signal[i]

    totsignal_val = tot_signal.values()

    # background
    bkg = [h[{"samples": label}] for label in bkg_labels]
    # sum all of the background
    if len(bkg) > 0:
        tot = bkg[0].copy()
        for i, b in enumerate(bkg):
            if i > 0:
                tot = tot + b

        tot_val = tot.values()
        tot_val_zero_mask = tot_val == 0
        tot_val[tot_val_zero_mask] = 1

    # replace values where bkg is 0
    totsignal_val[tot_val == 0] = 0

    # integrate soverb in a given range for lep_fj_m
    bin_array = tot_signal.axes[0].edges[
        :-1
    ]  # remove last element since bins have one extra element
    condition = (bin_array >= range_min) & (bin_array <= range_max)

    s = totsignal_val[condition].sum()
    b = np.sqrt(tot_val[condition].sum())

    soverb_integrated = round((s / b).item(), 2)
    print(
        f"S/sqrt(B) in range [{range_min}, {range_max}] of (Jet-Lep).mass is: {soverb_integrated}"
    )
    return soverb_integrated


def main(args):
    # append '_year' to the output directory
    odir = args.odir + "_" + args.year + "/stacked_hists/"

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

    for ch in channels:
        # load the hists
        with open(f"{odir}/{ch}_hists.pkl", "rb") as f:
            hists = pkl.load(f)
            f.close()
        print(f"Computing integrated soverb for {ch}...")
        compute_soverb(args.year, hists, ch)


if __name__ == "__main__":
    # e.g.
    # run locally as: python s_over_b.py --year 2017 --odir Nov11 --channels ele

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        dest="year",
        required=True,
        choices=["2016", "2016APV", "2017", "2018", "Run2"],
        help="year",
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

    args = parser.parse_args()

    main(args)
