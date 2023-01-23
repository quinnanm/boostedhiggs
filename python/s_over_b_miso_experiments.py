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
from s_over_b import compute_soverb

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


def make_hists(
    year, ch, idir, odir, low_pt_cuts, high_pt_cuts, weights, presel, samples
):
    """
    Makes 1D histograms to be plotted as stacked over the different samples
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        presel: pre-selection string
        weights: weights to be applied to MC
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json)
    """

    # get luminosity of year
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    f.close()
    print(
        f"Processing samples from year {year} with luminosity {luminosity} for channel {ch}"
    )

    # loop over the samples
    data_label = data_by_ch
    pt_iso = {"ele": 120, "mu": 55}

    hists = {}
    for low_pt_cut in low_pt_cuts:
        for high_pt_cut in high_pt_cuts:
            hists[(low_pt_cut, high_pt_cut)] = hist2.Hist(
                hist2.axis.StrCategory([], name="samples", growth=True),
                axis_dict["lep_fj_m"],
            )

    for sample in samples[year][ch]:
        print(f"Sample {sample}")
        # check if the sample was processed
        pkl_dir = f"{idir}_{year}/{sample}/outfiles/*.pkl"
        pkl_files = glob.glob(pkl_dir)
        if not pkl_files:  # skip samples which were not processed
            print(
                "- No processed files found...", pkl_dir, "skipping sample...", sample
            )
            continue

        # get list of parquet files that have been post processed
        parquet_files = glob.glob(f"{idir}_{year}/{sample}/outfiles/*_{ch}.parquet")

        # define an is_data boolean
        is_data = False
        for key in data_label.values():
            if key in sample:
                is_data = True

        # get combined sample
        sample_to_use = get_sample_to_use(sample, year)

        # get xsec_weight
        xsec_weight = get_xsecweight(pkl_files, year, sample, is_data, luminosity)

        for parquet_file in parquet_files:
            try:
                data = pq.read_table(parquet_file).to_pandas()
            except:
                if is_data:
                    print(
                        "Not able to read data: ",
                        parquet_file,
                        " should remove events from scaling/lumi",
                    )
                else:
                    print("Not able to read data from ", parquet_file)
                continue

            if len(data) == 0:
                print(
                    f"WARNING: Parquet file empty {year} {ch} {sample} {parquet_file}"
                )
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

            low_pt = data["lep_pt"] < pt_iso[ch]
            high_pt = data["lep_pt"] > pt_iso[ch]

            for low_pt_cut in low_pt_cuts:
                for high_pt_cut in high_pt_cuts:
                    con1 = low_pt & (data["lep_misolation"] < low_pt_cut)
                    con2 = high_pt & (data["lep_misolation"] < high_pt_cut)
                    condition = con1 | con2

                    hists[(low_pt_cut, high_pt_cut)].fill(
                        samples=sample_to_use,
                        var=data["lep_fj_m"][condition],
                        weight=event_weight[condition],
                    )

    # store the hists variable
    with open(f"{odir}/{ch}_hists.pkl", "wb") as f:
        pkl.dump(hists, f)


def main(args):
    # append '_year' to the output directory
    odir = args.odir + "_" + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + "/miso_hists/"):
        os.makedirs(odir + "/miso_hists/")
    odir = odir + "/miso_hists/"

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

    # list of weights to apply to MC
    weights = {}
    # pre-selection string
    presel = {}
    for ch in variables.keys():

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        presel_str = ""
        for sel in variables[ch]["pre-sel"]:
            presel_str += f"{sel} & "
        presel[ch] = presel_str[:-3]

    # cuts = [0.0001, 0.005, 0.010, 0.050, 0.1, 0.15, 0.2, 0.25, 0.3, 100]
    low_pt_cuts = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    high_pt_cuts = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    for ch in channels:
        if args.make_hists:
            if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
                print("Histograms already exist - remaking them")
            print(f"Making histograms for {ch}...")
            print("Weights: ", weights[ch])
            print("Pre-selection: ", presel[ch])
            make_hists(
                args.year,
                ch,
                args.idir,
                odir,
                low_pt_cuts,
                high_pt_cuts,
                weights[ch],
                presel[ch],
                samples,
            )
            print("-------------------------------------------")

        # load the hists
        with open(f"{odir}/{ch}_hists.pkl", "rb") as f:
            hists = pkl.load(f)
            f.close()

        soverb_all = []
        for low_pt_cut in low_pt_cuts:
            for high_pt_cut in high_pt_cuts:
                print(
                    f"Cut is lep_miso<{low_pt_cut} for low pT or lep_miso<{high_pt_cut} for high pT"
                )
                soverb_all.append(
                    compute_soverb(
                        args.year,
                        hists[(low_pt_cut, high_pt_cut)],
                        ch,
                        range_min=0,
                        range_max=150,
                        remove_ttH=False,
                    )
                )
                print("--------------------")


if __name__ == "__main__":
    # e.g.
    # run locally as: python s_over_b_miso_experiments.py --year 2017 --odir Nov11 --channels ele --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov4 --make_hists

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
        default="plot_configs/vars.json",
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
        "--make_hists", dest="make_hists", action="store_true", help="Make hists"
    )
    parser.add_argument(
        "--plot_hists", dest="plot_hists", action="store_true", help="Plot the hists"
    )
    parser.add_argument("--logy", dest="logy", action="store_true", help="Log y axis")
    parser.add_argument("--nodata", dest="nodata", action="store_false", help="No data")
    parser.add_argument(
        "--add_score", dest="add_score", action="store_true", help="Add inference score"
    )

    args = parser.parse_args()

    main(args)
