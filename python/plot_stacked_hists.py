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

cut_keys = [
    "trigger",
    "leptonKin",
    "fatjetKin",
    "ht",
    "oneLepton",
    "notaus",
    "leptonInJet",
    "pre-sel",
]

global axis_dict
axis_dict["cutflow"] = get_cutflow_axis(cut_keys)
print("Cutflow with key names: ", cut_keys)


def plot_stacked_hists(
    vars_to_plot, year, ch, odir, logy=True, add_data=True, add_soverb=True
):
    """
    Plots the stacked 1D histograms that were made by "make_hists" individually for each year
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    with open(f"{odir}/../{ch}_hists.pkl", "rb") as f:
        hists = pkl.load(f)
        f.close()

    odir += "/stacked_hists/"
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make the histogram plots in this directory
    if logy:
        if not os.path.exists(f"{odir}/{ch}_hists_log"):
            os.makedirs(f"{odir}/{ch}_hists_log")
    else:
        if not os.path.exists(f"{odir}/{ch}_hists"):
            os.makedirs(f"{odir}/{ch}_hists")

    # data label
    data_label = data_by_ch[ch]
    if year == "2018":
        data_label = data_by_ch_2018[ch]

    # luminosity
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[ch][year]
    luminosity = luminosity / 1000.0
    f.close()

    if year == "Run2":
        data_label = "Data"

    for var in vars_to_plot:
        if var not in hists.keys():
            continue

        if "gen" in var:
            continue

        print(var)

        # get histograms
        h = hists[var]

        if (
            h.shape[0] == 0
        ):  # skip empty histograms (such as lepton_pt for hadronic channel)
            print("Empty histogram ", var)
            continue

        # get samples existing in histogram
        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
        signal_labels = [label for label in samples if label in signal_by_ch[ch]]
        bkg_labels = [
            label
            for label in samples
            if (label and label != data_label and label not in signal_labels)
        ]

        # get total yield of backgrounds per label
        # (sort by yield in fixed fj_pt histogram after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            order_dic[simplified_labels[bkg_label]] = hists["fj_pt"][
                {"samples": bkg_label}
            ].sum()

        # data
        data = None
        if data_label in h.axes[0]:
            data = h[{"samples": data_label}]

        # signal
        signal = [h[{"samples": label}] for label in signal_labels]
        # scale signal for non-log plots
        if logy:
            mult_factor = 1
        else:
            mult_factor = 100
        signal_mult = [s * mult_factor for s in signal]

        # background
        bkg = [h[{"samples": label}] for label in bkg_labels]

        if add_data and data and len(bkg) > 0:
            if add_soverb and len(signal) > 0:
                fig, (ax, rax, sax) = plt.subplots(
                    nrows=3,
                    ncols=1,
                    figsize=(8, 8),
                    gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.07},
                    sharex=True,
                )
            else:
                fig, (ax, rax) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    figsize=(8, 8),
                    gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
                    sharex=True,
                )
                sax = None
        else:
            if add_soverb and len(signal) > 0:
                fig, (ax, sax) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    figsize=(8, 8),
                    gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
                    sharex=True,
                )
                rax = None
            else:
                fig, ax = plt.subplots(1, 1)
                rax = None
                sax = None

        errps = {
            "hatch": "////",
            "facecolor": "none",
            "lw": 0,
            "color": "k",
            "edgecolor": (0, 0, 0, 0.5),
            "linewidth": 0,
            "alpha": 0.4,
        }

        # sum all of the background
        if len(bkg) > 0:
            tot = bkg[0].copy()
            for i, b in enumerate(bkg):
                if i > 0:
                    tot = tot + b

            tot_val = tot.values()
            tot_val_zero_mask = tot_val == 0
            tot_val[tot_val_zero_mask] = 1

            tot_err = np.sqrt(tot_val)
            tot_err[tot_val_zero_mask] = 0

            # print(f'Background yield: ',tot_val,np.sum(tot_val))

        if add_data and data:
            data_err_opts = {
                "linestyle": "none",
                "marker": ".",
                "markersize": 10.0,
                "elinewidth": 1,
            }
            hep.histplot(
                data,
                ax=ax,
                histtype="errorbar",
                color="k",
                capsize=4,
                yerr=True,
                label=data_label,
                **data_err_opts,
            )

            if len(bkg) > 0:
                from hist.intervals import ratio_uncertainty

                data_val = data.values()
                data_val[tot_val_zero_mask] = 1

                yerr = ratio_uncertainty(data_val, tot_val, "poisson")
                # rax.stairs(
                #     1 + yerr[1],
                #     edges=tot.axes[0].edges,
                #     baseline=1 - yerr[0],
                #     **errps
                # )

                hep.histplot(
                    data_val / tot_val,
                    tot.axes[0].edges,
                    # yerr=np.sqrt(data_val) / tot_val,
                    yerr=yerr,
                    ax=rax,
                    histtype="errorbar",
                    color="k",
                    capsize=4,
                )

                rax.axhline(1, ls="--", color="k")
                rax.set_ylim(0.2, 1.8)
                # rax.set_ylim(0.7, 1.3)

        # plot the background
        if len(bkg) > 0:
            if var == "cutflow":
                """
                # sort bkg for cutflow
                summ = []
                for label in bkg_labels:
                    summ.append(order_dic[simplified_labels[label]])
                # get indices of labels arranged by yield
                order = []
                for i in range(len(summ)):
                    order.append(np.argmax(np.array(summ)))
                    summ[np.argmax(np.array(summ))] = -100
                bkg_ordered = [bkg[i] for i in order]
                bkg_labels_ordered = [bkg_labels[i] for i in order]
                """
                hep.histplot(
                    bkg,
                    ax=ax,
                    stack=True,
                    sort="yield",
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.5,
                    histtype="fill",
                    label=[simplified_labels[bkg_label] for bkg_label in bkg_labels],
                    color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
                )
            else:
                hep.histplot(
                    bkg,
                    ax=ax,
                    stack=True,
                    sort="yield",
                    edgecolor="black",
                    linewidth=1,
                    histtype="fill",
                    label=[simplified_labels[bkg_label] for bkg_label in bkg_labels],
                    color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
                )
            ax.stairs(
                values=tot.values() + tot_err,
                baseline=tot.values() - tot_err,
                edges=tot.axes[0].edges,
                **errps,
                label="Stat. unc.",
            )

        # plot the signal (times 10)
        if len(signal) > 0:
            tot_signal = None
            for i, sig in enumerate(signal_mult):
                lab_sig_mult = f"{mult_factor} * {simplified_labels[signal_labels[i]]}"
                if mult_factor == 1:
                    lab_sig_mult = f"{simplified_labels[signal_labels[i]]}"
                hep.histplot(
                    sig,
                    ax=ax,
                    label=lab_sig_mult,
                    linewidth=3,
                    color=color_by_sample[signal_labels[i]],
                )

                if tot_signal == None:
                    tot_signal = signal[i].copy()
                else:
                    tot_signal = tot_signal + signal[i]

            # plot the total signal (w/o scaling)
            hep.histplot(
                tot_signal, ax=ax, label=f"ggF+VBF+VH+ttH", linewidth=3, color="tab:red"
            )
            # add MC stat errors
            ax.stairs(
                values=tot_signal.values() + np.sqrt(tot_signal.values()),
                baseline=tot_signal.values() - np.sqrt(tot_signal.values()),
                edges=sig.axes[0].edges,
                **errps,
            )

            if sax is not None:
                totsignal_val = tot_signal.values()
                # replace values where bkg is 0
                totsignal_val[tot_val == 0] = 0
                soverb_val = totsignal_val / np.sqrt(tot_val)
                hep.histplot(
                    soverb_val,
                    tot_signal.axes[0].edges,
                    label="Total Signal",
                    ax=sax,
                    linewidth=3,
                    color="tab:red",
                )

                # integrate soverb in a given range for lep_fj_m (which, intentionally, is the first variable we pass)
                if var == "lep_fj_m":
                    bin_array = tot_signal.axes[0].edges[
                        :-1
                    ]  # remove last element since bins have one extra element
                    range_max = 150
                    range_min = 0

                    condition = (bin_array >= range_min) & (bin_array <= range_max)

                    s = totsignal_val[
                        condition
                    ].sum()  # sum/integrate signal counts in the range
                    b = np.sqrt(
                        tot_val[condition].sum()
                    )  # sum/integrate bkg counts in the range and take sqrt

                    soverb_integrated = round((s / b).item(), 2)
                    sax.legend(title=f"S/sqrt(B) (in 0-150)={soverb_integrated}")

        ax.set_ylabel("Events")
        if sax is not None:
            ax.set_xlabel("")
            if rax is not None:
                rax.set_xlabel("")
                rax.set_ylabel("Data/MC", fontsize=20)
            sax.set_ylabel(r"S/$\sqrt{B}$", fontsize=20)
            sax.set_xlabel(f"{axis_dict[var].label}")
            if var == "cutflow":
                sax.set_xticks(range(len(cut_keys)), cut_keys, rotation=60)

        elif rax is not None:
            ax.set_xlabel("")
            rax.set_xlabel(f"{axis_dict[var].label}")
            rax.set_ylabel("Data/MC", fontsize=20)
            if var == "cutflow":
                rax.set_xticks(range(len(cut_keys)), cut_keys, rotation=60)
        else:
            if var == "cutflow":
                ax.set_xticks(range(len(cut_keys)), cut_keys, rotation=60)

        # get handles and labels of legend
        handles, labels = ax.get_legend_handles_labels()

        # append legend labels in order to a list
        summ = []
        for label in labels[: len(bkg_labels)]:
            summ.append(order_dic[label])
        # get indices of labels arranged by yield
        order = []
        for i in range(len(summ)):
            order.append(np.argmax(np.array(summ)))
            summ[np.argmax(np.array(summ))] = -100

        # plot data first, then bkg, then signal
        hand = [handles[-1]] + [handles[i] for i in order] + handles[len(bkg) : -1]
        lab = [labels[-1]] + [labels[i] for i in order] + labels[len(bkg) : -1]

        ax.legend(
            [hand[idx] for idx in range(len(hand))],
            [lab[idx] for idx in range(len(lab))],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title=f"{label_by_ch[ch]} Channel",
        )

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.1)

        hep.cms.lumitext(
            "%.1f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20
        )
        hep.cms.text("Work in Progress", ax=ax, fontsize=15)

        if logy:
            # print(f"Saving to {odir}/{ch}_hists_log/{var}.pdf")
            # plt.savefig(f"{odir}/{ch}_hists_log/{var}.pdf", bbox_inches="tight")
            plt.savefig(f"{odir}/{ch}_hists_log/{var}.png", bbox_inches="tight")
        else:
            # print(f"Saving to {odir}/{ch}_hists/{var}.pdf")
            # plt.savefig(f"{odir}/{ch}_hists/{var}.pdf", bbox_inches="tight")
            plt.savefig(f"{odir}/{ch}_hists/{var}.png", bbox_inches="tight")
        plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)
    odir = odir + "/stacked_hists/"
    if not os.path.exists(odir):
        os.makedirs(odir)

    channels = args.channels.split(",")

    vars_to_plot = {}
    if args.var is not None:
        for ch in channels:
            vars_to_plot[ch] = args.var.split(",")
    else:
        with open(args.vars) as f:
            variables = yaml.safe_load(f)
        for ch in variables.keys():
            vars_to_plot[ch] = []
            for key, value in variables[ch]["vars"].items():
                if value == 1:
                    vars_to_plot[ch].append(key)

    for ch in channels:
        print(f"Plotting for {ch}...")
        plot_stacked_hists(
            vars_to_plot[ch], args.year, ch, odir, logy=True, add_data=args.nodata
        )
        plot_stacked_hists(
            vars_to_plot[ch], args.year, ch, odir, logy=False, add_data=args.nodata
        )


if __name__ == "__main__":
    # e.g.
    # run locally as: python plot_stacked_hists.py --year 2017 --odir Nov15 --channels ele
    # run locally as: python plot_stacked_hists.py --year 2017 --odir Nov23tagger --channels ele,mu

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
    parser.add_argument("--var", dest="var", default=None, help="variable to plot")
    parser.add_argument("--nodata", dest="nodata", action="store_false", help="No data")

    args = parser.parse_args()

    main(args)
