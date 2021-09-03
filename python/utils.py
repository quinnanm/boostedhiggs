import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

def data_label(region):
    if region=="hadel":
        datalabel="SingleElectron"
    elif region=="hadmu":
        datalabel="SingleMuon"
    else:
        datalabel="Data"
    return datalabel

def plot_cutflow(data, sig, bkg, bkg_labels=None, region="hadel", odir="./", year=None):
    regions = {
            "hadel": ["none", "triggere", "met_filters", "lep_in_fj", "fjmsd", "oneelectron", "el_iso", "btag_ophem_med", "mt_lep_met"],
            "hadmu": ["none", "triggermu", "met_filters", "lep_in_fj", "fjmsd", "onemuon", "mu_iso", "btag_ophem_med", "mt_lep_met"],
        }

    fig, ax = plt.subplots(
        figsize=(12,10),
        tight_layout=True,
    )

    hep.histplot(
        [bkg.values() for bkg in bkg],
        ax=ax,
        stack=True,
        histtype="fill",
        label=bkg_labels
    )
    hep.histplot(
        sig.values(),
        ax=ax,
        color="cyan",
        label="HWW"
    )

    datalabel = data_label(region)

    hep.histplot(
        data.values(),
        ax=ax,
        histtype="errorbar",
        color="k",
        yerr=True,
        label=datalabel,
    )
    plt.xticks(
        ticks=np.arange(len(regions[region])),
        labels=regions[region],
    )
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    plt.ylabel("Events")
    plt.xlim(0,9)
    plt.legend(frameon=True)
    plt.gca().set_yscale("log")
    plt.gcf().savefig("%s/%s_%s_cutflow.png"%(odir,region,year))

hist_labels = {
    'mt': r"$m_T(l, p_T^{miss})$ [GeV]"
}

def plot_stack(data, sig, bkg, bkg_labels, sig_label="HWW (1700)", histname=None,
               region="hadel", odir="./", year=None):
    """
    data: hist
    sig: hist
    bkg: hist
    """
    
    plt.rcParams.update({
        'font.size': 14,
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
        1700*sig,
        ax=ax,
        color="cyan",
        label=sig_label,
    )

    # real data
    datalabel =data_label(region)
    hep.histplot(
        data,
        ax=ax,
        histtype="errorbar",
        color="k",
        yerr=True,
        label=datalabel,
    )

    # ratio plot
    rax.errorbar(
        x=[data.axes.value(i)[0] for i in range(len(data.values()))],
        y=data.values() / sig.values(),
        fmt="ko",
        #yerr=np.sqrt(data.values() / sig.values()),
    )

    # axes labels and limits
    ax.set(
        ylabel="Events",
        xlabel=None,
    )
    ax.legend(
        loc="best",
        frameon=True,
    )
    rax.set(
        xlabel=hist_labels[histname],
        ylabel="Data/Pred",
        ylim=(0,2)
    )
    rax.yaxis.label.set_size(18)

    # save fig
    fig.savefig("%s/%s_%s_%s.png"%(odir,region,year,histname))
