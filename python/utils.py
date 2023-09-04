#!/usr/bin/python

import json
import os
import pickle as pkl
import warnings

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")


combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # signal
    "GluGluHToWW_Pt-200ToInf_M-125": "HWW",
    "HToWW_M-125": "VH",
    "VBFHToWWToLNuQQ_M-125_withDipoleRecoil": "VBF",
    "ttHToNonbb_M125": "ttH",
    # bkg
    "QCD_Pt": "QCD",
    "DYJets": "DYJets",
    "WJetsToLNu_": "WJetsLNu",
    "JetsToQQ": "WZQQ",
    "TT": "TTbar",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "GluGluHToTauTau": "HTauTau",
}
signals = ["HWW", "ttH", "VH", "VBF"]


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
def disc_score(df, sigs, bkgs):
    num = df[sigs].sum(axis=1)
    den = df[sigs].sum(axis=1) + df[bkgs].sum(axis=1)
    return num / den


# signal scores definition
hwwev = ["fj_PN_probHWqqWev0c", "fj_PN_probHWqqWev1c", "fj_PN_probHWqqWtauev0c", "fj_PN_probHWqqWtauev1c"]
hwwmv = ["fj_PN_probHWqqWmv0c", "fj_PN_probHWqqWmv1c", "fj_PN_probHWqqWtaumv0c", "fj_PN_probHWqqWtaumv1c"]
hwwhad = [
    "fj_PN_probHWqqWqq0c",
    "fj_PN_probHWqqWqq1c",
    "fj_PN_probHWqqWqq2c",
    "fj_PN_probHWqqWq0c",
    "fj_PN_probHWqqWq1c",
    "fj_PN_probHWqqWq2c",
    "fj_PN_probHWqqWtauhv0c",
    "fj_PN_probHWqqWtauhv1c",
]
sigs = hwwev + hwwmv + hwwhad

# background scores definition
qcd = ["fj_PN_probQCDbb", "fj_PN_probQCDcc", "fj_PN_probQCDb", "fj_PN_probQCDc", "fj_PN_probQCDothers"]

tope = ["fj_PN_probTopbWev", "fj_PN_probTopbWtauev"]
topm = ["fj_PN_probTopbWmv", "fj_PN_probTopbWtaumv"]
tophad = ["fj_PN_probTopbWqq0c", "fj_PN_probTopbWqq1c", "fj_PN_probTopbWq0c", "fj_PN_probTopbWq1c", "fj_PN_probTopbWtauhv"]
top = tope + topm + tophad

# use ParT
new_sig = [s.replace("PN", "ParT") for s in sigs]
qcd_bkg = [b.replace("PN", "ParT") for b in qcd]
top_bkg = [b.replace("PN", "ParT") for b in tope + topm + tophad]
inclusive_bkg = [b.replace("PN", "ParT") for b in qcd + tope + topm + tophad]
# ---------------------------------------------------------

# PLOTTING UTILS
color_by_sample = {
    "HWW": "pink",
    "VH": "tab:brown",
    "VBF": "tab:gray",
    "ttH": "tab:olive",
    "DYJets": "tab:purple",
    "QCD": "tab:orange",
    "Diboson": "orchid",
    "WJetsLNu": "tab:green",
    "TTbar": "tab:blue",
    "WZQQ": "salmon",
    "SingleTop": "tab:cyan",
    #     "WplusHToTauTau": "tab:cyan",
    #     "WminusHToTauTau": "tab:cyan",
    #     "ttHToTauTau": "tab:cyan",
    #     "GluGluHToTauTau": "tab:cyan",
    #     "ZHToTauTau": "tab:cyan",
    #     "VBFHToTauTau": "tab:cyan",
}

plot_labels = {
    "HWW": "ggH(WW)-Pt200",
    "VH": "VH(WW)",
    "VBF": r"VBFH(WW) $(qq\ell\nu)$",
    "ttH": "ttH(WW)",
    "DYJets": r"Z$(\ell\ell)$+jets",
    "QCD": "Multijet",
    "Diboson": "VV",
    "WJetsLNu": r"W$(\ell\nu)$+jets",
    "TTbar": r"$t\bar{t}$+jets",
    "WZQQ": r"W/Z$(qq)$",
    "SingleTop": r"Single Top",
    #     "WplusHToTauTau": "WplusHToTauTau",
    #     "WminusHToTauTau": "WminusHToTauTau",
    #     "ttHToTauTau": "ttHToTauTau",
    #     "GluGluHToTauTau": "GluGluHToTauTau",
    #     "ZHToTauTau": "ZHToTauTau",
    #     "VBFHToTauTau": "VBFHToTauTau"
}

sig_labels = {
    "HWW": "ggF",
    "VBF": "VBF",
    "VH": "VH",
    "ttH": "ttH",
}

label_by_ch = {"mu": "Muon", "ele": "Electron"}

axis_dict = {
    "Zmass": hist2.axis.Regular(40, 30, 450, name="var", label=r"Zmass [GeV]", overflow=True),
    "ht": hist2.axis.Regular(30, 200, 1200, name="var", label=r"ht [GeV]", overflow=True),
    "lep_pt": hist2.axis.Regular(40, 30, 450, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "met": hist2.axis.Regular(40, 0, 450, name="var", label=r"MET", overflow=True),
    "fj_minus_lep_m": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "fj_minus_lep_pt": hist2.axis.Regular(40, 0, 450, name="var", label=r"Jet - Lepton $p_T$ [GeV]", overflow=True),
    "lep_met_mt": hist2.axis.Regular(35, 0, 400, name="var", label=r"$m_T(lep, p_T^{miss})$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "lep_fj_dr": hist2.axis.Regular(35, 0.0, 0.8, name="var", label=r"$\Delta R(Jet, Lepton)$", overflow=True),
    # "lep_fj_dr": hist2.axis.Regular(35, 0.8, 5, name="var", label=r"$\Delta R(Jet, Lepton)$", overflow=True),
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 200, 600, name="var", label=r"Jet $p_T$ [GeV]", overflow=True),
    "fj_msoftdrop": hist2.axis.Regular(35, 20, 250, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
    "rec_higgs_m": hist2.axis.Regular(35, 0, 480, name="var", label=r"Higgs reconstructed mass [GeV]", overflow=True),
    "rec_higgs_pt": hist2.axis.Regular(30, 0, 1000, name="var", label=r"Higgs reconstructed $p_T$ [GeV]", overflow=True),
    "fj_pt_over_lep_pt": hist2.axis.Regular(35, 1, 10, name="var", label=r"$p_T$(Jet) / $p_T$(Lepton)", overflow=True),
    "rec_higgs_pt_over_lep_pt": hist2.axis.Regular(
        35, 1, 10, name="var", label=r"$p_T$(Recontructed Higgs) / $p_T$(Lepton)", overflow=True
    ),
    "golden_var": hist2.axis.Regular(35, 0, 10, name="var", label=r"$p_{T}(W_{l\nu})$ / $p_{T}(W_{qq})$", overflow=True),
    "rec_dphi_WW": hist2.axis.Regular(
        35, 0, 3.14, name="var", label=r"$\left| \Delta \phi(W_{l\nu}, W_{qq}) \right|$", overflow=True
    ),
    "fj_ParT_mass": hist2.axis.Regular(20, 0, 250, name="var", label=r"ParT regressed mass [GeV]", overflow=True),
    "fj_ParticleNet_mass": hist2.axis.Regular(
        35, 0, 250, name="var", label=r"fj_ParticleNet regressed mass [GeV]", overflow=True
    ),
    "met_fj_dphi": hist2.axis.Regular(
        35, 0, 3.14, name="var", label=r"$\left| \Delta \phi(MET, Jet) \right|$", overflow=True
    ),
    # VBF
    "deta": hist2.axis.Regular(35, 0, 7, name="var", label=r"$\left| \Delta \eta_{jj} \right|$", overflow=True),
    "mjj": hist2.axis.Regular(35, 0, 2000, name="var", label=r"$m_{jj}$", overflow=True),
    "fj_genjetpt": hist2.axis.Regular(30, 200, 600, name="var", label=r"Gen Jet $p_T$ [GeV]", overflow=True),
    "jet_resolution": hist2.axis.Regular(
        30, -3, 3, name="var", label=r"(Gen Jet $p_T$ - Jet $p_T$)/Gen Jet $p_T$", overflow=True
    ),
    "nj": hist2.axis.Regular(40, 0, 10, name="var", label="number of jets outside candidate jet", overflow=True),
    "inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
    "fj_ParT_inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
    "fj_ParT_all_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
}


def plot_hists(years, channels, hists, vars_to_plot, add_data, logy, add_soverb, only_sig, mult, outpath):
    # luminosity
    luminosity = 0
    for year in years:
        lum = 0
        for ch in channels:
            with open("../fileset/luminosity.json") as f:
                lum += json.load(f)[ch][year] / 1000.0

        luminosity += lum / len(channels)

    for var in vars_to_plot:
        if var not in hists.keys():
            print(f"{var} not stored in hists")
            continue

        print(f"Will plot {var} histogram")

        # get histograms
        h = hists[var]

        if h.shape[0] == 0:  # skip empty histograms (such as lepton_pt for hadronic channel)
            print("Empty histogram ", var)
            continue

        # get samples existing in histogram
        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
        signal_labels = [label for label in samples if label in signals]
        bkg_labels = [label for label in samples if (label and label not in signal_labels and (label not in ["Data"]))]

        # get total yield of backgrounds per label
        # (sort by yield in fixed fj_pt histogram after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            if "fj_pt" in hists.keys():
                order_dic[plot_labels[bkg_label]] = hists["fj_pt"][{"samples": bkg_label}].sum()
            else:
                order_dic[plot_labels[bkg_label]] = hists[var][{"samples": bkg_label}].sum()

        # data
        if add_data:
            data = h[{"samples": "Data"}]

        # signal
        signal = [h[{"samples": label}] for label in signal_labels]
        # scale signal for non-log plots
        if logy:
            mult_factor = 1
        else:
            mult_factor = mult
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
            hep.histplot(
                data,
                ax=ax,
                histtype="errorbar",
                color="k",
                capsize=4,
                yerr=True,
                label="Data",
                **data_err_opts,
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
                lab_sig_mult = f"{mult_factor} * {plot_labels[signal_labels[i]]}"
                if mult_factor == 1:
                    lab_sig_mult = f"{plot_labels[signal_labels[i]]}"
                hep.histplot(
                    sig,
                    ax=ax,
                    label=lab_sig_mult,
                    linewidth=3,
                    color=color_by_sample[signal_labels[i]],
                )

                if tot_signal is None:
                    tot_signal = signal[i].copy()
                else:
                    tot_signal = tot_signal + signal[i]

            # plot the total signal (w/o scaling)
            hep.histplot(tot_signal, ax=ax, label="ggF+VBF+VH+ttH", linewidth=3, color="tab:red")
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

                # integrate soverb in a given range for fj_minus_lep_mass
                if var == "fj_minus_lep_m":
                    bin_array = tot_signal.axes[0].edges[:-1]  # remove last element since bins have one extra element
                    range_max = 150
                    range_min = 0

                    condition = (bin_array >= range_min) & (bin_array <= range_max)

                    s = totsignal_val[condition].sum()  # sum/integrate signal counts in the range
                    b = np.sqrt(tot_val[condition].sum())  # sum/integrate bkg counts in the range and take sqrt

                    soverb_integrated = round((s / b).item(), 2)
                    sax.legend(title=f"S/sqrt(B) (in 0-150)={soverb_integrated:.2f}")
                # integrate soverb in a given range for rec_higgs_m
                if var == "rec_higgs_m":
                    bin_array = tot_signal.axes[0].edges[:-1]  # remove last element since bins have one extra element
                    range_max = 200
                    range_min = 50

                    condition = (bin_array >= range_min) & (bin_array <= range_max)

                    s = totsignal_val[condition].sum()  # sum/integrate signal counts in the range
                    b = np.sqrt(tot_val[condition].sum())  # sum/integrate bkg counts in the range and take sqrt

                    soverb_integrated = round((s / b).item(), 2)
                    sax.legend(title=f"S/sqrt(B) (in 50-200)={soverb_integrated:.2f}")

        ax.set_ylabel("Events")
        if sax is not None:
            ax.set_xlabel("")
            if rax is not None:
                rax.set_xlabel("")
                rax.set_ylabel("Data/MC", fontsize=20)
            sax.set_ylabel(r"S/$\sqrt{B}$", fontsize=20)
            sax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

        elif rax is not None:
            ax.set_xlabel("")
            rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis

            rax.set_ylabel("Data/MC", fontsize=20)

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

        if len(channels) == 2:
            ax.legend(
                [hand[idx] for idx in range(len(hand))],
                [lab[idx] for idx in range(len(lab))],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="Semi-Leptonic Channel",
            )
        else:
            ax.legend(
                [hand[idx] for idx in range(len(hand))],
                [lab[idx] for idx in range(len(lab))],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title=f"{label_by_ch[ch]} Channel",
            )

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1)

        hep.cms.lumitext("%.1f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
        hep.cms.text("Work in Progress", ax=ax, fontsize=15)

        # save plot
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        plt.savefig(f"{outpath}/{var}.pdf", bbox_inches="tight")
