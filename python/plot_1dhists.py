#!/usr/bin/python

from utils import (
    axis_dict,
    add_samples,
    color_by_sample,
    signal_by_ch,
    simplified_labels,
    color_by_sample,
    get_cutflow_axis,
)
import pickle as pkl
import os
import sys
import argparse

import hist as hist2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings

warnings.filterwarnings("ignore", message="Found duplicate branch ")

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})


def plot_1dhists(year, channels, odir, var, samples, tag, logy):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
    """

    # load the hists
    hists = {}
    for ch in channels:
        with open(f"{odir}/../{ch}_hists.pkl", "rb") as f:
            hists[ch] = pkl.load(f)
            f.close()

    print(hists[channels[0]].keys())

    # make plots per channel
    try:
        h = hists[channels[0]][var]
    except:
        print(f"Variable {var} not present in hists pkl file")
        exit

    # print(h.axes[0].edges)
    all_samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
    # print(all_samples)

    # get samples: hists[channels[0]][var].axes[1]

    ch_titles = {"ele": "Electron", "mu": "Muon", "all": "Semi-leptonic"}
    for ch in channels:
        ch_title = ch_titles[ch]
        fig, ax = plt.subplots(figsize=(8, 8))
        for sample in samples:
            try:
                h = hists[ch][var][{"samples": sample}]
            except:
                raise Exception("Unable to access histogram - samples available ",hists[ch][var][{"var": sum}])
            try:
                label = simplified_labels[sample]
            except:
                raise Exception("Unable to access simplified_label")
            try:
                color = color_by_sample[sample]
            except:
                raise Exception("Unable to access color_by_sample")
            
            hep.histplot(
                h,
                ax=ax,
                label=label,
                linewidth=3,
                color=color,
            )
            
        hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        ax.grid(linestyle='-', linewidth=0.2)
        ax.legend(title=f"{ch_title} Channel")
        if var == "cutflow":
            ax.set_xticks(range(len(cut_keys)), cut_keys, rotation=40, fontsize=13)
            ax.set_ylabel("Events (~normalized to XS - nogenweight)")
            ax.set_xlabel("")
        if logy:
            ax.set_ylim(10, 50000)
            ax.set_yscale("log")
            plt.savefig(f"{odir}/1dhist_{var}_{tag}_{ch}_log.pdf")
        else:
            plt.savefig(f"{odir}/1dhist_{var}_{tag}_{ch}.pdf")
        plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        print(f"Output directory {odir} with histograms does not exist")
        exit

    # make subdirectory specefic to this script
    if not os.path.exists(odir + "/1d_hists/"):
        os.makedirs(odir + "/1d_hists/")
    odir = odir + "/1d_hists/"

    channels = args.channels.split(",")

    print(f"Plotting...")
    plot_1dhists(
        args.year,
        channels,
        odir,
        args.var,
        args.samples.split(","),
        args.tag,
        args.logy,
    )


if __name__ == "__main__":
    # e.g. run locally as
    # python plot_1dhists.py --year 2017 --odir hists --channels ele --var lep_pt
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", dest="samples", required=True, help="string w samples"
    )
    parser.add_argument("--tag", dest="tag", required=True, help="tag the plot")
    parser.add_argument("--year", dest="year", default="2017", help="year")
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
        help="tag for output directory... will append '/{year}' to it",
    )
    parser.add_argument(
        "--var", dest="var", default=None, required=True, help="variable to plot"
    )
    parser.add_argument("--logy", dest="logy", action="store_true", help="Log y axis")
    parser.add_argument(
        "--cut-keys",
        dest="cut_keys",
        default="trigger,leptonKin,fatjetKin,ht,oneLepton,notaus,leptonInJet,pre-sel",
        help="cut keys for cutflow (split by commas)",
    )

    args = parser.parse_args()

    cut_keys = args.cut_keys.split(",")
    global axis_dict
    axis_dict["cutflow"] = get_cutflow_axis(cut_keys)

    main(args)
