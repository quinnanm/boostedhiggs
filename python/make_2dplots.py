#!/usr/bin/python

from utils import data_by_ch, data_by_ch_2018
from utils import get_sample_to_use, get_xsecweight

import yaml
import pickle as pkl
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import scipy
import json
import os, sys, glob
import argparse

import hist as hist2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_2dplots(ch, idir, odir, weights, presel, samples, y_start, y_end):
    """
    Makes 2D plots of two variables

    Args:
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    # scores = ["hww_score (H)", "H/(1-H)", "H/(H+QCD)", "H(H+Top)"]#, "qcd_score (QCD)", "top_score (Top)"]
    scores = ["hww_score (H)"]

    # instantiates the histogram object
    hists = {}
    for score in scores:
        hists[score] = hist2.Hist(
            hist2.axis.StrCategory([], name='samples', growth=True),
            hist2.axis.Regular(20, 0, 4, name="miso", label="miso", overflow=True),
            hist2.axis.Regular(50, y_start, y_end, name="score", label=score, overflow=True),
        )

    labels = []
    # loop over the samples
    for yr in samples.keys():

        # data label and lumi
        data_label = data_by_ch[ch]
        if yr == "2018":
            data_label = data_by_ch_2018[ch]
        f = open("../fileset/luminosity.json")
        luminosity = json.load(f)[ch][yr]
        f.close()
        print(f"Processing samples from year {yr} with luminosity {luminosity} for channel {ch}")

        for sample in samples[yr][ch]:
            if "ttHToNonbb_M125" in sample: # skip ttH cz labels are not stored
                continue 
            if data_label in sample:    # skip data samples
                continue
            is_data = False

            print(f"Sample {sample}")

            # check if the sample was processed
            pkl_dir = f"{idir}_{yr}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)
            if not pkl_files:  # skip samples which were not processed
                print("- No processed files found...", pkl_dir, "skipping sample...", sample)
                continue

            # get list of parquet files that have been post processed
            parquet_files = glob.glob(f"{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet")

            # get combined sample
            sample_to_use = get_sample_to_use(sample,yr)

            # get cutflow
            xsec_weight = get_xsecweight(pkl_files,yr,sample,is_data,luminosity)

            for parquet_file in parquet_files:
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    print("Not able to read data from ", parquet_file)
                    continue

                # retrieve tagger labels from one parquet
                if len(labels)==0:
                    for key in data.keys():
                        if "label" in key:
                            labels.append(key)
                            print(key)

                if len(data) == 0:
                    print(f"WARNING: Parquet file empty {yr} {ch} {sample} {parquet_file}")
                    continue

                # modify dataframe with pre-selection query
                if presel is not None:
                    data = data.query(presel)

                event_weight = xsec_weight
                weight_ones = np.ones_like(data["weight_genweight"])
                for w in weights:
                    try:
                        event_weight *= data[w]
                    except:
                        if w!="weight_vjets_nominal":
                            print(f"No {w} variable in parquet for sample {sample}")
      
                # apply softmax per row
                df_all_softmax = scipy.special.softmax(data[labels].values, axis=1)

                # combine the labels under QCD, TOP or HWW
                QCD = 0
                TOP = 0
                HIGGS = 0

                for i, label in enumerate(labels):  # labels are the tagger classes
                    if "QCD" in label:
                        QCD += df_all_softmax[:, i]
                    elif "Top" in label:
                        TOP += df_all_softmax[:, i]
                    else:
                        HIGGS += df_all_softmax[:, i]

                # filling the scores nd arrays
                for score in scores:
                    if score=="hww_score (H)":
                        X = np.ndarray.tolist(HIGGS)
                    elif score=="H/(1-H)":
                        X = np.ndarray.tolist(HIGGS/(1-HIGGS))
                    elif score=="H/(H+QCD)":
                        X = np.ndarray.tolist(HIGGS/(HIGGS+QCD))
                    elif score=="H/(H+Top)":
                        X = np.ndarray.tolist(HIGGS/(HIGGS+TOP))

                    hists[score].fill(
                        samples=sample_to_use,
                        miso=data["lep_misolation"],
                        score=X,
                        weight=event_weight,
                    )   

    # store the hists variable
    with open(f"{odir}/{ch}_2d_scores.pkl", "wb") as f:
        pkl.dump(hists, f)


def plot_2dplots(year, ch, odir):
    """
    Plots 2D plots of two variables that were made by "make_2dplots" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    with open(f'{odir}/{ch}_2d_scores.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    save_dict = {"hww_score (H)": "H",
                "H/(1-H)": "H_normalized",
                "H/(H+QCD)": "H_normalized_QCD",
                "H(H+Top)": "H_normalized_Top"
                }

    for score in hists.keys():
        # make directory to store stuff per year
        if not os.path.exists(f'{odir}/{ch}_{save_dict[score]}'):
            os.makedirs(f'{odir}/{ch}_{save_dict[score]}')

        # make plots per channel
        for sample in hists[score].axes[0]:
            print(sample)
            # one for log z-scale
            # fig, ax = plt.subplots(figsize=(8, 5))
            # hep.hist2dplot(hists[score][{'samples': sample}], ax=ax, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=10000))
            # ax.set_xlabel(f"miso")
            # ax.set_ylabel(f"{score}")
            # ax.set_title(f'{ch} channel for \n {sample}')
            # hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
            # hep.cms.text("Work in Progress", ax=ax)
            # print(f'saving at {odir}/{ch}_{save_dict[score]}/{sample}_log_z.png')
            # plt.savefig(f'{odir}/{ch}_{save_dict[score]}/{sample}_log_z.png')
            # plt.close()

            # one for non-log z-scale
            fig, ax = plt.subplots(figsize=(8, 5))
            hep.hist2dplot(hists[score][{'samples': sample}], ax=ax, cmap="plasma", cmin=0, cmax=1)
            ax.set_xlabel(f"miso")
            ax.set_ylabel(f"{score}")
            ax.set_title(f'{ch} channel for \n {sample}')
            hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
            hep.cms.text("Work in Progress", ax=ax)
            print(f'saving at {odir}/{ch}_{save_dict[score]}/{sample}.png')
            plt.savefig(f'{odir}/{ch}_{save_dict[score]}/{sample}.png')
            plt.close()


def main(args):

    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        os.system(f'mkdir -p {odir}')

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

    # load yaml config file
    with open(args.vars) as f:
        variables = yaml.safe_load(f)

    print(variables)

    # list of weights to apply to MC
    weights = {}
    # pre-selection string
    presel = {}
    for ch in channels:

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        presel_str = None
        if type(variables[ch]["pre-sel"]) is list:
            presel_str = variables[ch]["pre-sel"][0]
            for i,sel in enumerate(variables[ch]["pre-sel"]):
                if i==0: continue
                presel_str += f'& {sel}'
        presel[ch] = presel_str

    os.system(f"cp {args.vars} {odir}/")

    for ch in channels:
        samples[args.year][ch] = []
        for key, value in json_samples[args.year][ch].items():
            if value == 1:
                samples[args.year][ch].append(key)

    for ch in channels:

        if args.make_hists:
                print(f'Making 2dplot of tagger scores vs miso for {ch} channel')
                make_2dplots(ch, args.idir, odir, weights[ch], presel[ch], samples, args.y_start, args.y_end)

        if args.plot_hists:
            print('Plotting...')
            plot_2dplots(args.year, ch, odir)


if __name__ == "__main__":
    # e.g. run locally as
    # python make_2dplots.py --year 2017 --odir Dec15_2d_plots --channels ele --make_hists --plot_hists --y_start 0 --y_end 1 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov16_inference

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                                 help="year")
    parser.add_argument(
        "--samples",
        dest="samples",
        default="plot_configs/samples_pfnano.json",
        help="path to json with samples to be plotted",
    )
    parser.add_argument('--channels',        dest='channels',    default='ele',                                  help="channel... choices are ['ele', 'mu', 'had']")
    parser.add_argument('--odir',            dest='odir',        default='hists',                                help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                          help="input directory with results")
    parser.add_argument(
        "--vars", dest="vars", default="plot_configs/vars.yaml", help="path to json with variables to be plotted"
    )
    parser.add_argument('--y_bins',          dest='y_bins',      default=50,                                     help="binning of the second variable passed",               type=int)
    parser.add_argument('--y_start',         dest='y_start',     default=0,                                      help="starting range of the second variable passed",        type=float)
    parser.add_argument('--y_end',           dest='y_end',       default=1,                                      help="end range of the second variable passed",             type=float)
    parser.add_argument("--make_hists",      dest='make_hists',  action='store_true',                            help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',  action='store_true',                            help="Plot the hists")

    args = parser.parse_args()

    main(args)
