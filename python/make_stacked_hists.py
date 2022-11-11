from utils import axis_dict, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
from utils import simplified_labels, get_cutflow, get_xsecweight, get_sample_to_use, get_cutflow_axis

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
plt.rcParams.update({'font.size': 20})

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
print('Cutflow with key names: ',cut_keys)

def make_hists(year, ch, idir, odir, vars_to_plot, weights, presel, samples):
    """
    Makes 1D histograms to be plotted as stacked over the different samples
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        vars_to_plot: the set of variables to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
        presel: pre-selection string
        weights: weights to be applied to MC
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json)
    """
    
    # get luminosity of year
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    f.close()
    print(f"Processing samples from year {year} with luminosity {luminosity} for channel {ch}")

    # define histograms
    hists = {}
    sample_axis = hist2.axis.StrCategory([], name="samples", growth=True)
    plot_vars = vars_to_plot
    plot_vars.append("cutflow")
    for var in plot_vars:
        hists[var] = hist2.Hist(
            sample_axis,
            axis_dict[var],
        )

    # cutflow dictionary
    cut_values = {}

    # pt cuts for variables
    pt_iso = {"ele": 120, "mu": 55}

    # loop over the samples
    for yr in samples.keys():
        if yr == "2018":
            data_label = data_by_ch_2018
        else:
            data_label = data_by_ch

        for sample in samples[yr][ch]:
            print(f"Sample {sample}")
            # check if the sample was processed
            pkl_dir = f"{idir}_{yr}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)
            if not pkl_files:  # skip samples which were not processed
                print("- No processed files found...", pkl_dir, "skipping sample...", sample)
                continue

            # get list of parquet files that have been post processed
            parquet_files = glob.glob(f"{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet")

            # define an is_data boolean
            is_data = False
            for key in data_label.values():
                if key in sample:
                    is_data = True

            # get combined sample
            sample_to_use = get_sample_to_use(sample,yr)

            # get cutflow
            xsec_weight = get_xsecweight(pkl_files,yr,sample,is_data,luminosity)
            if sample_to_use not in cut_values.keys():
                cut_values[sample_to_use] = dict.fromkeys(cut_keys,0)
            cutflow = get_cutflow(cut_keys,pkl_files,yr,sample,xsec_weight,ch)
            for key,val in cutflow.items():
                cut_values[sample_to_use][key] += val

            cut_values[sample_to_use]["pre-sel"] = 0

            sample_yield = 0
            for parquet_file in parquet_files:
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    if is_data:
                        print("Not able to read data: ", parquet_file, " should remove events from scaling/lumi")
                    else:
                        print("Not able to read data from ", parquet_file)
                    continue

                if len(data) == 0:
                    print(f"WARNING: Parquet file empty {yr} {ch} {sample} {parquet_file}")
                    continue

                # modify dataframe with pre-selection query
                data = data.query(presel)

                if not is_data:
                    event_weight = xsec_weight
                    weight_ones = np.ones_like(data["weight_genweight"])
                    for w in weights:
                        try:
                            event_weight *= data[w]
                        except:
                            print("No {w} variable in parquet")
                            # break
                else:
                    weight_ones = np.ones_like(data["lep_pt"])
                    event_weight = weight_ones
                    
                # account yield for extra selection (with only the xsec weight)
                sample_yield += np.sum(weight_ones*xsec_weight)

                for var in plot_vars:
                    if var == "cutflow":
                        continue
                    if var == "score" and not args.add_score:
                        continue

                    # for specific variables introduce cut
                    if "lowpt" in var:
                        select_var = (data["lep_pt"] < pt_iso[ch])
                    elif "highpt" in var:
                        select_var = (data["lep_pt"] > pt_iso[ch])
                    else:
                        select_var = (data["lep_pt"] > 0)
                        
                    var_plot = var.replace('_lowpt', '').replace('_highpt', '')
                    if var_plot not in data.keys():
                        print(f"Var {var} not in parquet keys")
                        continue

                    # filling histograms
                    hists[var].fill(
                        samples=sample_to_use,
                        var=data[var_plot][select_var],
                        weight=event_weight[select_var],
                    )

            cut_values[sample_to_use]["pre-sel"] += sample_yield

            # fill cutflow histogram once we have all the values
            for key, numevents in cut_values[sample_to_use].items():
                cut_index = cut_keys.index(key)
                hists["cutflow"].fill(
                    samples=sample_to_use,
                    var=cut_index,
                    weight=numevents
                )

            #samples = [hists["cutflow"].axes[0].value(i) for i in range(len(hists["cutflow"].axes[0].edges))]
            #print(sample,samples)

        # save cutflow values
        with open(f"{odir}/cut_values_{ch}.pkl", "wb") as f:
            pkl.dump(cut_values, f)

    samples = [hists["cutflow"].axes[0].value(i) for i in range(len(hists["cutflow"].axes[0].edges))]
    print(samples)

    # store the hists variable
    with open(f"{odir}/{ch}_hists.pkl", "wb") as f:
        pkl.dump(hists, f)


def plot_stacked_hists(year, ch, odir, vars_to_plot, logy=True, add_data=True, add_soverb=True):
    """
    Plots the stacked 1D histograms that were made by "make_hists" individually for each year
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        vars_to_plot: the set of variable to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
    """

    # load the hists
    with open(f"{odir}/{ch}_hists.pkl", "rb") as f:
        hists = pkl.load(f)
        f.close()

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
    elif year == "Run2":
        data_label = "Data"

    # luminosity
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    luminosity = luminosity / 1000.
    f.close()


    for var in vars_to_plot[ch]:
        print(var)

        # get histograms
        h = hists[var]

        if h.shape[0] == 0:  # skip empty histograms (such as lepton_pt for hadronic channel)
            print("Empty histogram ", var)
            continue

        # get samples existing in histogram
        samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]
        signal_labels = [label for label in samples if label in signal_by_ch[ch]]
        bkg_labels = [label for label in samples if (label and label != data_label and label not in signal_labels)]

        # get total yield of backgrounds per label
        # (sort by yield in fixed fj_pt histogram after pre-sel)
        order_dic = {}
        for bkg_label in bkg_labels:
            order_dic[simplified_labels[bkg_label]] = hists["fj_pt"][{"samples": bkg_label}].sum()

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
                    nrows=3, ncols=1, figsize=(8, 8), gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.07}, sharex=True
                )
            else:
                fig, (ax, rax) = plt.subplots(
                    nrows=2, ncols=1, figsize=(8, 8), gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07}, sharex=True
                )
                sax = None
        else:
            if add_soverb and len(signal) > 0:
                fig, (ax, sax) = plt.subplots(
                    nrows=2, ncols=1, figsize=(8, 8), gridspec_kw={"height_ratios": (4, 1), "hspace": 0.07}, sharex=True
                )
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
            tot_val_zero_mask = (tot_val == 0)
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
                data, ax=ax, histtype="errorbar", color="k", capsize=4, yerr=True, label=data_label, **data_err_opts
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
                    #yerr=np.sqrt(data_val) / tot_val,
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
                label='Stat. unc.'
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
                tot_signal,
                ax=ax,
                label=f"ggF+VBF+VH+ttH",
                linewidth=3,
                color='tab:red'
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
                totsignal_val[tot_val==0] = 0
                soverb_val = totsignal_val / np.sqrt(tot_val)
                hep.histplot(
                    soverb_val,
                    tot_signal.axes[0].edges,
                    label='Total Signal',
                    ax=sax,
                    linewidth=3,
                    color='tab:red',
                )
                # integrate soverb in a given range for lep_fj_m (which, intentionally, is the first variable we pass)
                if var=="lep_fj_m":
                    
                    bin_array = tot_signal.axes[0].edges[:-1] # remove last element since bins have one extra element
                    range_max = 150
                    range_min = 0

                    condition = (bin_array>=range_min) & (bin_array<=range_max)
            
                    s = totsignal_val[condition].sum()  # sum/integrate signal counts in the range
                    b = np.sqrt(tot_val[condition].sum())   # sum/integrate bkg counts in the range and take sqrt

                    soverb_integrated = round((s/b).item(),2)

                sax.legend(title=f"S/sqrt(B) (in 0-150)={soverb_integrated}")

        ax.set_ylabel("Events")
        if sax is not None:
            ax.set_xlabel("")
            rax.set_xlabel("")
            sax.set_ylabel("S/sqrt(B)",fontsize=20)
            sax.set_xlabel(f"{axis_dict[var].label}")
            rax.set_ylabel("Data/MC",fontsize=20)
            if var=="cutflow":
                sax.set_xticks(range(len(cut_keys)), cut_keys, rotation=60)

        elif rax is not None:
            ax.set_xlabel("")
            rax.set_xlabel(f"{axis_dict[var].label}")
            rax.set_ylabel("Data/MC",fontsize=20)
            if var=="cutflow":
                rax.set_xticks(range(len(cut_keys)), cut_keys, rotation=60)
        else:
            if var=="cutflow":
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
        hand = [handles[-1]] + [handles[i] for i in order] + handles[len(bkg):-1]
        lab = [labels[-1]] + [labels[i] for i in order] + labels[len(bkg):-1]

        ax.legend(
            [hand[idx] for idx in range(len(hand))], 
            [lab[idx] for idx in range(len(lab))], bbox_to_anchor=(1.05, 1), 
            loc='upper left', title=f"{label_by_ch[ch]} Channel"
        )
        
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.1)

        hep.cms.lumitext("%.1f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
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
    odir = args.odir + "_" + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + "/stacked_hists/"):
        os.makedirs(odir + "/stacked_hists/")
    odir = odir + "/stacked_hists/"

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

    f = open(args.vars)
    variables = json.load(f)
    f.close()

    # variables to plot
    vars_to_plot = {}
    # list of weights to apply to MC
    weights = {}
    # pre-selection string
    presel = {}
    for ch in variables.keys():
        vars_to_plot[ch] = []
        for key, value in variables[ch]["vars"].items():
            if value == 1:
                vars_to_plot[ch].append(key)

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        presel_str = ""
        for sel in variables[ch]["pre-sel"]:
            presel_str += f"{sel} & "
        presel[ch] = presel_str[:-3]
    
    for ch in channels:
        if args.make_hists:
            if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
                print("Histograms already exist - remaking them")
            print(f"Making histograms for {ch}...")
            print("Weights: ",weights[ch])
            print("Pre-selection: ",presel[ch])
            make_hists(args.year, ch, args.idir, odir, vars_to_plot[ch], weights[ch], presel[ch], samples)

        if args.plot_hists:
            print(f"Plotting for {ch}...")
            plot_stacked_hists(args.year, ch, odir, vars_to_plot, logy=args.logy, add_data=args.nodata)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_stacked_hists.py --year 2017 --odir Nov10 --channels ele --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov4 --plot_hists --make_hists

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", dest="year", required=True, choices=["2016", "2016APV", "2017", "2018", "Run2"], help="year"
    )
    parser.add_argument(
        "--vars", dest="vars", default="plot_configs/vars.json", help="path to json with variables to be plotted"
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        default="plot_configs/samples_pfnano.json",
        help="path to json with samples to be plotted",
    )
    parser.add_argument("--channels", dest="channels", default="ele,mu", help="channels for which to plot this variable")
    parser.add_argument(
        "--odir", dest="odir", default="hists", help="tag for output directory... will append '_{year}' to it"
    )
    parser.add_argument("--idir", dest="idir", default="../results/", help="input directory with results - without _{year}")
    parser.add_argument("--make_hists", dest="make_hists", action="store_true", help="Make hists")
    parser.add_argument("--plot_hists", dest="plot_hists", action="store_true", help="Plot the hists")
    parser.add_argument("--logy", dest="logy", action="store_true", help="Log y axis")
    parser.add_argument("--nodata", dest="nodata", action="store_false", help="No data")
    parser.add_argument("--add_score", dest="add_score",  action="store_true", help="Add inference score")

    args = parser.parse_args()

    main(args)
