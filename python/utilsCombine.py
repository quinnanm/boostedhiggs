#!/usr/bin/python

import json
import os
import pickle as pkl
import warnings

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import onnx
import onnxruntime as ort
import scipy

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")


# (name of sample, name in templates)
combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # signal
    "GluGluHToWW_Pt-200ToInf_M-125": "ggF",
    "VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil": "VBF",
    # "VBFHToWWToLNuQQ_M-125_withDipoleRecoil": "VBF",
    "ttHToNonbb_M125": "ttH",
    "HWminusJ_HToWW_M-125": "WH",
    "HWplusJ_HToWW_M-125": "WH",
    "HZJ_HToWW_M-125": "ZH",
    "GluGluZH_HToWW_M-125_TuneCP5_13TeV-powheg-pythia8": "ZH",
    # bkg
    "QCD_Pt": "QCD",
    "DYJets": "DYJets",
    "WJetsToLNu_": "WJetsLNu",
    "TT": "TTbar",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "JetsToQQ": "WZQQ",
    "EWK": "EWKvjets",
    # "GluGluHToTauTau": "HTauTau",
}

signals = ["ggF", "VBF"]


def get_sum_sumgenweight(pkl_files, year, sample):
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


# ---------------------------------------------------------
# TAGGER STUFF
def get_finetuned_score(data, modelv="v2_nor2"):
    # add finetuned tagger score
    PATH = f"../../weaver-core-dev/experiments_finetuning/{modelv}/model.onnx"

    input_dict = {
        "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
    }

    onnx_model = onnx.load(PATH)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(
        PATH,
        providers=["AzureExecutionProvider"],
    )
    outputs = ort_sess.run(None, input_dict)

    return scipy.special.softmax(outputs[0], axis=1)[:, 0]


# ---------------------------------------------------------

# PLOTTING UTILS
color_by_sample = {
    # signal
    "ggF": "pink",
    "VBF": "tab:orange",
    # higgs background
    "WH": "tab:brown",
    "ZH": "tab:grey",
    "ttH": "tab:olive",
    # background
    "QCD": "tab:orange",
    "DYJets": "tab:purple",
    "WJetsLNu": "tab:green",
    "TTbar": "tab:blue",
    "SingleTop": "tab:cyan",
    "Diboson": "orchid",
    "WZQQ": "salmon",
    "EWKvjets": "grey",
    #     "WplusHToTauTau": "tab:cyan",
    #     "WminusHToTauTau": "tab:cyan",
    #     "ttHToTauTau": "tab:cyan",
    #     "GluGluHToTauTau": "tab:cyan",
    #     "ZHToTauTau": "tab:cyan",
    #     "VBFHToTauTau": "tab:cyan",
}

plot_labels = {
    # signal
    "ggF": "ggH",
    "VBF": "VBF",
    "ttH": "ttH",
    "WH": "WH",
    "ZH": "ZH",
    # background
    "QCD": "Multijet",
    "DYJets": r"Z$(\ell\ell)$+jets",
    "WJetsLNu": r"W$(\ell\nu)$+jets",
    "TTbar": r"$t\bar{t}$+jets",
    "SingleTop": r"Single Top",
    "Diboson": "VV",
    "WZQQ": r"W/Z$(qq)$",
    "EWKvjets": r"EWKVJets$",
    #     "WplusHToTauTau": "WplusHToTauTau",
    #     "WminusHToTauTau": "WminusHToTauTau",
    #     "ttHToTauTau": "ttHToTauTau",
    #     "GluGluHToTauTau": "GluGluHToTauTau",
    #     "ZHToTauTau": "ZHToTauTau",
    #     "VBFHToTauTau": "VBFHToTauTau"
}

label_by_ch = {"mu": "Muon", "ele": "Electron"}


def plot_hists(
    hists,
    years,
    channels,
    add_data,
    logy,
    add_soverb,
    only_sig,
    mult,
    outpath,
    text_="",
    blind_region=None,
    save_as=None,
):
    # luminosity
    luminosity = 0
    for year in years:
        lum = 0
        for ch in channels:
            with open("../fileset/luminosity.json") as f:
                lum += json.load(f)[ch][year] / 1000.0

        luminosity += lum / len(channels)

    # get histograms
    h = hists

    if h.shape[0] == 0:  # skip empty histograms (such as lepton_pt for hadronic channel)
        print("Empty histogram")
        return None

    # get samples existing in histogram
    samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
    signal_labels = [label for label in samples if label in signals]
    bkg_labels = [label for label in samples if (label and label not in signal_labels and (label not in ["Data"]))]

    # get total yield of backgrounds per label
    # (sort by yield in fixed fj_pt histogram after pre-sel)
    order_dic = {}
    for bkg_label in bkg_labels:
        order_dic[plot_labels[bkg_label]] = hists[{"Sample": bkg_label}].sum()

    # data
    if add_data:
        data = h[{"Sample": "Data"}]

    # signal
    signal = [h[{"Sample": label}] for label in signal_labels]
    # scale signal for non-log plots
    if logy:
        mult_factor = 1
    else:
        mult_factor = mult
    signal_mult = [s * mult_factor for s in signal]

    # background
    bkg = [h[{"Sample": label}] for label in bkg_labels]

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
            fig, ax = plt.subplots(figsize=(8, 8))
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

    if add_data and data:
        data_err_opts = {
            "linestyle": "none",
            "marker": ".",
            "markersize": 10.0,
            "elinewidth": 1,
        }

        if blind_region:
            massbins = data.axes[-1].edges
            lv = int(np.searchsorted(massbins, blind_region[0], "right"))
            rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)

            data.view(flow=True)[lv:rv].value = 0
            data.view(flow=True)[lv:rv].variance = 0

        hep.histplot(
            data,
            ax=ax,
            histtype="errorbar",
            color="k",
            capsize=4,
            yerr=True,
            label="Data",
            **data_err_opts,
            flow="none",
        )

        if len(bkg) > 0:
            from hist.intervals import ratio_uncertainty

            data_val = data.values()
            data_val[tot_val_zero_mask] = 1

            yerr = ratio_uncertainty(data_val, tot_val, "poisson")

            hep.histplot(
                data_val / tot_val,
                tot.axes[0].edges,
                yerr=yerr,
                ax=rax,
                histtype="errorbar",
                color="k",
                capsize=4,
                flow="none",
            )

            rax.axhline(1, ls="--", color="k")
            rax.set_ylim(0.2, 1.8)

    # plot the background
    if len(bkg) > 0 and not only_sig:
        hep.histplot(
            bkg,
            ax=ax,
            stack=True,
            sort="yield",
            edgecolor="black",
            linewidth=1,
            histtype="fill",
            label=[plot_labels[bkg_label] for bkg_label in bkg_labels],
            color=[color_by_sample[bkg_label] for bkg_label in bkg_labels],
            flow="none",
        )
        ax.stairs(
            values=tot.values() + tot_err,
            baseline=tot.values() - tot_err,
            edges=tot.axes[0].edges,
            **errps,
            label="Stat. unc.",
        )

    # ax.text(0.5, 0.9, text_, fontsize=14, transform=ax.transAxes, weight="bold")

    # plot the signal (times 10)
    if len(signal) > 0:
        tot_signal = None
        for i, sig in enumerate(signal_mult):
            lab_sig_mult = f"{mult_factor} * {plot_labels[signal_labels[i]]}"
            if mult_factor == 1:
                lab_sig_mult = f"{plot_labels[signal_labels[i]]}"
            hep.histplot(
                sig,
                ax=ax,
                label=lab_sig_mult,
                linewidth=3,
                color=color_by_sample[signal_labels[i]],
                flow="none",
            )

            if tot_signal is None:
                tot_signal = signal[i].copy()
            else:
                tot_signal = tot_signal + signal[i]

        # plot the total signal (w/o scaling)
        hep.histplot(tot_signal, ax=ax, label="ggF+VBF+VH+ttH", linewidth=3, color="tab:red", flow="none")
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
                flow="none",
            )

            bin_array = tot_signal.axes[0].edges[:-1]  # remove last element since bins have one extra element
            range_max = 160
            range_min = 80

            condition = (bin_array >= range_min) & (bin_array <= range_max)

            s = totsignal_val[condition].sum()  # sum/integrate signal counts in the range
            b = np.sqrt(tot_val[condition].sum())  # sum/integrate bkg counts in the range and take sqrt

            soverb_integrated = round((s / b).item(), 2)
            # sax.legend(title=f"S/sqrt(B) (in {range_min}-{range_max})={soverb_integrated:.2f}")

            if "SR1" in text_:
                sax.set_ylim(0, 1.7)
                sax.set_yticks([0, 0.5, 1.0, 1.5])
            elif "SR2" in text_:
                sax.set_ylim(0, 0.7)
                sax.set_yticks([0, 0.2, 0.4, 0.6])

            sax.set_ylim(0, 1.2)
            sax.set_yticks([0, 0.5, 1.0])

            # sax.set_ylim(0, 0.5)
            # sax.set_yticks([0, 0.3])
            # sax.set_ylim(0, 0.5)
            # sax.set_yticks([0, 0.2, 0.4])

    ax.set_ylabel("Events")
    if sax is not None:
        ax.set_xlabel("")
        if rax is not None:
            rax.set_xlabel("")
            rax.set_ylabel("Data/MC", fontsize=20)
        sax.set_ylabel(r"S/$\sqrt{B}$", fontsize=20, y=0.4, labelpad=0)
        sax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

    elif rax is not None:
        ax.set_xlabel("")
        rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

        rax.set_ylabel("Data/MC", fontsize=20, labelpad=0)

    # get handles and labels of legend
    handles, labels = ax.get_legend_handles_labels()

    # append legend labels in order to a list
    summ = []
    for label in labels[: len(bkg_labels)]:
        summ.append(order_dic[label].value)
    # get indices of labels arranged by yield
    order = []
    for i in range(len(summ)):
        order.append(np.argmax(np.array(summ)))
        summ[np.argmax(np.array(summ))] = -100

    # plot data first, then bkg, then signal
    hand = [handles[-1]] + [handles[i] for i in order] + handles[len(bkg) : -1]
    lab = [labels[-1]] + [labels[i] for i in order] + labels[len(bkg) : -1]

    if len(channels) == 1:
        if channels[0] == "ele":
            text_ += " electron"
        else:
            text_ += " muon"

    ax.legend(
        [hand[idx] for idx in range(len(hand))],
        [lab[idx] for idx in range(len(lab))],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title=text_,
    )

    # if len(channels) == 2:
    #     ax.legend(
    #         [hand[idx] for idx in range(len(hand))],
    #         [lab[idx] for idx in range(len(lab))],
    #         bbox_to_anchor=(1.05, 1),
    #         loc="upper left",
    #         title="Semi-Leptonic Channel",
    #     )
    # else:
    #     ax.legend(
    #         [hand[idx] for idx in range(len(hand))],
    #         [lab[idx] for idx in range(len(lab))],
    #         bbox_to_anchor=(1.05, 1),
    #         loc="upper left",
    #         title=f"{label_by_ch[ch]} Channel",
    #     )

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-1)
    else:
        ax.set_ylim(0)
    ax.set_xlim(45, 210)
    hep.cms.lumitext("%.0f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
    hep.cms.text("Work in Progress", ax=ax, fontsize=15)

    # save plot
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if save_as:
        plt.savefig(f"{outpath}/{save_as}_stacked_hists.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"{outpath}/stacked_hists.pdf", bbox_inches="tight")
