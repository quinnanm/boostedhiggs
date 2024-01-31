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


combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # signal
    "GluGluHToWW_Pt-200ToInf_M-125": "ggF",
    "HToWW_M-125": "VH",
    "VBF": "VBF",
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
    "EWK": "EWKvjets",
}
signals = ["ggF", "ttH", "VH", "VBF"]


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
    "ggF": "pink",
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
    # "WJetsLNu_unmatched": "tab:grey",
    # "WJetsLNu_matched": "tab:green",
    "EWKvjets": "tab:grey",
}

plot_labels = {
    "ggF": "ggF",
    "VH": "VH",
    # "VH": "VH(WW)",
    # "VBF": r"VBFH(WW) $(qq\ell\nu)$",
    "VBF": r"VBF",
    # "ttH": "ttH(WW)",
    "ttH": r"$t\bar{t}$H",
    "DYJets": r"Z$(\ell\ell)$+jets",
    "QCD": "Multijet",
    "Diboson": "VV",
    "WJetsLNu": r"W$(\ell\nu)$+jets",
    "TTbar": r"$t\bar{t}$+jets",
    "WZQQ": r"V$(qq)$",
    "SingleTop": r"Single T",
    #     "WplusHToTauTau": "WplusHToTauTau",
    #     "WminusHToTauTau": "WminusHToTauTau",
    #     "ttHToTauTau": "ttHToTauTau",
    #     "GluGluHToTauTau": "GluGluHToTauTau",
    #     "ZHToTauTau": "ZHToTauTau",
    #     "VBFHToTauTau": "VBFHToTauTau"
    "WJetsLNu_unmatched": r"W$(\ell\nu)$+jets unmatched",
    "WJetsLNu_matched": r"W$(\ell\nu)$+jets matched",
    "EWKvjets": "EWK VJets",
}

# sig_labels = {
#     "HWW": "ggF",
#     "VBF": "VBF",
#     "VH": "VH",
#     "ttH": "ttH",
# }

label_by_ch = {"mu": "Muon", "ele": "Electron"}

axis_dict = {
    "Zmass": hist2.axis.Regular(40, 30, 450, name="var", label=r"Zmass [GeV]", overflow=True),
    "fj_minus_lep_m": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "fj_minus_lep_pt": hist2.axis.Regular(40, 0, 450, name="var", label=r"Jet - Lepton $p_T$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "rec_higgs_etajet_m": hist2.axis.Variable(
        list(range(50, 240, 20)), name="var", label=r"PKU definition Higgs reconstructed mass [GeV]", overflow=True
    ),
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
    # VBF
    "deta": hist2.axis.Regular(35, 0, 7, name="var", label=r"$\left| \Delta \eta_{jj} \right|$", overflow=True),
    "mjj": hist2.axis.Regular(35, 0, 2000, name="var", label=r"$m_{jj}$", overflow=True),
    "fj_genjetpt": hist2.axis.Regular(30, 200, 600, name="var", label=r"Gen Jet $p_T$ [GeV]", overflow=True),
    "jet_resolution": hist2.axis.Regular(
        30, -3, 3, name="var", label=r"(Gen Jet $p_T$ - Jet $p_T$)/Gen Jet $p_T$", overflow=True
    ),
    "nj": hist2.axis.Regular(40, 0, 10, name="var", label="number of jets outside candidate jet", overflow=True),
    "inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
    "fj_ParT_score_finetuned": hist2.axis.Regular(25, 0.5, 1, name="var", label=r"$T_{HWW}$", overflow=True),
    "fj_ParT_inclusive_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"ParT-Finetuned score", overflow=True),
    "fj_ParT_all_score": hist2.axis.Regular(35, 0, 1, name="var", label=r"tagger score", overflow=True),
    ############# AN
    "FirstFatjet_pt": hist2.axis.Regular(30, 250, 600, name="var", label=r"Leading AK8 jet $p_T$ [GeV]", overflow=True),
    "SecondFatjet_pt": hist2.axis.Regular(30, 250, 600, name="var", label=r"Sub-Leading AK8 jet $p_T$ [GeV]", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 250, 600, name="var", label=r"Higgs candidate jet $p_T$ [GeV]", overflow=True),
    "lep_pt": hist2.axis.Regular(40, 30, 400, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "NumFatjets": hist2.axis.Regular(5, 0.5, 5.5, name="var", label="Number of AK8 jets", overflow=True),
    "NumOtherJets": hist2.axis.Regular(
        7, 0.5, 7.5, name="var", label="Number of AK4 jets (non-overlapping with AK8)", overflow=True
    ),
    "lep_fj_dr": hist2.axis.Regular(
        35, 0.03, 0.8, name="var", label=r"$\Delta R(\ell, \mathrm{Higgs \ candidate \ jet})$", overflow=True
    ),
    "met_pt": hist2.axis.Regular(40, 20, 250, name="var", label=r"MET [GeV]", overflow=True),
    "met_fj_dphi": hist2.axis.Regular(
        35, 0, 1.57, name="var", label=r"$\left| \Delta \phi(MET, \mathrm{Higgs \ candidate \ jet}) \right|$", overflow=True
    ),
    "lep_met_mt": hist2.axis.Regular(35, 0, 160, name="var", label=r"$m_T(\ell, p_T^{miss})$ [GeV]", overflow=True),
    "ht": hist2.axis.Regular(30, 400, 1400, name="var", label=r"ht [GeV]", overflow=True),
    "rec_W_qq_m": hist2.axis.Regular(40, 0, 160, name="var", label=r"Reconstructed $W_{qq}$ mass [GeV]", overflow=True),
    "rec_W_lnu_m": hist2.axis.Regular(
        40, 0, 160, name="var", label=r"Reconstructed $W_{\ell \nu}$ mass [GeV]", overflow=True
    ),
    "fj_msoftdrop": hist2.axis.Regular(35, 20, 200, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
    "fj_lsf3": hist2.axis.Regular(35, 0, 1, name="var", label=r"Jet lsf3", overflow=True),
    # lepton isolation
    "lep_isolation": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Lepton PF isolation", overflow=True),
    "lep_isolation_ele": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Electron PF isolation", overflow=True),
    "lep_isolation_ele_highpt": hist2.axis.Regular(
        35, 0, 0.5, name="var", label=r"Electron PF isolation (high $p_T$)", overflow=True
    ),
    "lep_isolation_ele_lowpt": hist2.axis.Regular(
        35, 0, 0.15, name="var", label=r"Electron PF isolation (low $p_T$)", overflow=True
    ),
    "lep_isolation_mu": hist2.axis.Regular(35, 0, 0.5, name="var", label=r"Muon PF isolation", overflow=True),
    "lep_isolation_mu_highpt": hist2.axis.Regular(
        35, 0, 0.5, name="var", label=r"Muon PF isolation (high $p_T$)", overflow=True
    ),
    "lep_isolation_mu_lowpt": hist2.axis.Regular(
        35, 0, 0.15, name="var", label=r"Muon PF isolation (low $p_T$)", overflow=True
    ),
    "lep_misolation": hist2.axis.Regular(35, 0, 0.2, name="var", label=r"Muon mini-isolation", overflow=True),
    "lep_misolation_highpt": hist2.axis.Regular(
        35, 0, 0.2, name="var", label=r"Muon mini-isolation (high $p_T$)", overflow=True
    ),
    "lep_misolation_lowpt": hist2.axis.Regular(
        35, 0, 0.2, name="var", label=r"Muon mini-isolation (low $p_T$)", overflow=True
    ),
}


def plot_hists(
    years,
    channels,
    hists,
    vars_to_plot,
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

        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07},
            sharex=True,
        )

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

            if blind_region and ("rec_higgs" in var):
                massbins = data.axes[-1].edges
                lv = int(np.searchsorted(massbins, blind_region[0], "right"))
                rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)

                data.view(flow=True)[lv:rv] = 0

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
                if tot_signal is None:
                    tot_signal = signal[i].copy()
                else:
                    tot_signal = tot_signal + signal[i]

            # plot the total signal (w/o scaling)
            tot_signal *= mult_factor
            hep.histplot(
                tot_signal,
                ax=ax,
                label=r"ggF+VBF+VH+ttH $\times$" + f"{mult_factor}",
                linewidth=2,
                color="tab:red",
                flow="none",
            )
            # add MC stat errors
            ax.stairs(
                values=tot_signal.values() + np.sqrt(tot_signal.values()),
                baseline=tot_signal.values() - np.sqrt(tot_signal.values()),
                edges=sig.axes[0].edges,
                **errps,
            )

        ax.set_ylabel("Events")

        ax.set_xlabel("")
        rax.set_xlabel(f"{h.axes[-1].label}")  # assumes the variable to be plotted is at the last axis
        rax.set_ylabel("Data/MC", fontsize=20, labelpad=10)

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

        # plot bkg, then signal, then data
        hand = [handles[i] for i in order] + handles[len(bkg) : -1] + [handles[-1]]
        lab = [labels[i] for i in order] + labels[len(bkg) : -1] + [labels[-1]]

        lab_new, hand_new = [], []
        for i in range(len(lab)):
            if "Stat" in lab[i]:
                continue

            lab_new.append(lab[i])
            hand_new.append(hand[i])

        if len(channels) == 1:
            if channels[0] == "ele":
                chlab = text_ + " electron"
            else:
                chlab = text_ + " muon"
        else:
            chlab = text_

        ax.legend(
            [hand_new[idx] for idx in range(len(hand_new))],
            [lab_new[idx] for idx in range(len(lab_new))],
            ncol=2
            #             bbox_to_anchor=(1.05, 1),
            #             loc="upper left",
        )

        _, a = ax.get_ylim()
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1, a * 1.5)
        else:
            ax.set_ylim(0, a * 1.5)

        if "Num" in var:
            ax.set_xlim(h.axes["var"].edges[0], h.axes["var"].edges[-1])
        else:
            ax.set_xlim(h.axes["var"].edges[0], h.axes["var"].edges[-1])

        hep.cms.lumitext("%.0f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
        hep.cms.text("Work in Progress", ax=ax, fontsize=15)

        # save plot
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if save_as:
            plt.savefig(f"{outpath}/{save_as}_stacked_hists_{var}.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"{outpath}/stacked_hists_{var}.pdf", bbox_inches="tight")
