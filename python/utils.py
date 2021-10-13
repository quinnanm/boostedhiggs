import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

axis_labels = {
    "jetpt": r"Jet $p_T$ [GeV]",
    "jetmsd": r"Jet $m_{sd} [GeV]$",
    "jetrho": r"$\rho$",
    "btag": "btagFlavB (opphem)",
    "met": r"$p_T^{miss}$ [GeV]",
    "mt_lepmet": r"$m_T(lep, p_T^{miss})$ [GeV]",
    "lepminiIso": r"lep miniIso",
    "leprelIso": "lep Rel Iso",
    "lep_pt": r'lep $p_T$ [GeV]',
    "deltaR_lepjet": r"$\Delta R(l, Jet)$"
}

axis_limits = {
    "jetpt": (None, 800),
    "jetmsd": (None, None),
    "jetrho": (-6, -1),
    "btag": (-0.02, 0.3),
    "met": (None, None),
    "mt_lepmet": (None, None),
    "lepminiIso": (None, None),
    "leprelIso": (None, None),
    "lep_pt": (None, None),
    "deltaR_lepjet": (None, None),
}


def data_label(region):
    if region=="hadel":
        datalabel="SingleElectron"
    elif region=="hadmu":
        datalabel="SingleMuon"
    else:
        datalabel="Data"
    return datalabel


def plot_cutflow(data, sig, bkg, bkg_labels=None, region="hadel", odir="./", year=2017):
    regions = {
            "hadel": ["none", "triggere", "met_filters", "oneelectron", "fjacc", "fjmsd", "btag_ophem_med", "met_20", "lep_in_fj", "mt_lep_met",\
 "el_iso"],
            "hadmu": ["none", "triggermu", "met_filters", "onemuon", "fjacc", "fjmsd", "btag_ophem_med", "met_20", "lep_in_fj", "mt_lep_met", "mu_iso"],
}

    fig, ax = plt.subplots(
        figsize=(12,12),
        tight_layout=True,
    )

    hep.histplot(
        [bkg.values() for bkg in bkg],
        ax=ax,
        stack=True,
        edgecolor="k",
        histtype="fill",
        label=bkg_labels
    )
    hep.histplot(
        sig.values(),
        ax=ax,
        color="cyan",
        label="H(WW)"
    )

    hep.histplot(
        data.values(),
        ax=ax,
        histtype="errorbar",
        color="k",
        yerr=True,
        label=data_label(region),
    )
    # axes labels, limits and ticks
    plt.xticks(
        ticks=np.arange(len(regions[region])),
        labels=regions[region],
    )
    plt.setp(
        ax.xaxis.get_majorticklabels(), 
        rotation=90, 
        ha="left"
    )
    ax.set(
        ylabel="Events",
        xlim=(0, len(regions[region])),
        yscale="log"
    )
    ax.legend(
        frameon=True, 
        prop={"size":15}, 
        loc="lower left"
    )

    hep.cms.lumitext("2017 (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)
    
    #save fig
    fig.savefig(f"{odir}/{region}_{year}_cutflow.png")


def plot_stack(data, sig, bkg, bkg_labels, sig_label="HWW (1000)", 
               axis_name=None, region="hadel", odir="./", year=2017):

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8,8),
        tight_layout=True,
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    hep.histplot(
        bkg,
        ax=ax,
        stack=True,
        histtype="fill",
        edgecolor="k",
        alpha=0.8,
        label=bkg_labels,
    )
    hep.histplot(
        1000*sig,
        ax=ax,
        color="cyan",
        label=sig_label,
    )

    # real data
    hep.histplot(
        data,
        ax=ax,
        histtype="errorbar",
        color="k",
        yerr=True,
        label=data_label(region),
    )

    # ratio plot
    rax.errorbar(
        x=[data.axes.value(i)[0] for i in range(len(data.values()))],
        y=data.values() / np.sum([b.values() for b in bkg], axis=0),
        fmt="ko",
    )

    # axes labels and limits
    ax.set(
        ylabel="Events",
        xlabel=None,
        xlim=axis_limits[axis_name]
    )
    ax.legend(
        loc="best",
        frameon=True,
    )
    rax.set(
        xlabel=axis_labels[axis_name],
        xlim=axis_limits[axis_name],
        ylabel="Data/Background",
        ylim=(0,2)
    )
    rax.yaxis.label.set_size(13)

    hep.cms.lumitext("2017 (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)

    # save fig
    fig.savefig(f"{odir}/{region}_{year}_{axis_name}.png")
