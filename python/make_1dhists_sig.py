#!/usr/bin/python

import warnings
from hist.intervals import clopper_pearson_interval
from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
from utils import get_simplified_label, get_sum_sumgenweight, simplified_labels
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import sys
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})


warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_1dhists(year, ch, idir, odir, samples, var, bins, range):
    """
    Makes 1D histograms

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
    """

    # instantiates the histogram object
    hists = hist2.Hist(
        hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),     # to combine different pt bins of the same process
    )

    # loop over the samples
    for sample in samples[year][ch]:
        print("------------------------------------------------------------")
        # check if the sample was processed
        pkl_dir = f'{idir}/{sample}/outfiles/*.pkl'
        pkl_files = glob.glob(pkl_dir)  #
        if not pkl_files:  # skip samples which were not processed
            print('- No processed files found...', pkl_dir, 'skipping sample...', sample)
            continue
        print(f"Processing sample {sample}")

        # check if the sample was processed
        parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')

        if len(parquet_files) != 0:
            print(f'Processing {ch} channel of sample', sample)

        for i, parquet_file in enumerate(parquet_files):
            try:
                data = pq.read_table(parquet_file).to_pandas()
            except:
                print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                continue
            if len(data) == 0:
                continue

            try:
                data[var]
            except:
                print(f"{sample} doesn't have {var} stored")
                continue    # if sample doesn't have stored variable

            single_sample = None
            for single_key, key in add_samples.items():
                if key in sample:
                    single_sample = single_key

            if single_sample is not None:
                hists.fill(
                    data[var],
                    single_sample,  # combining all events under one name
                )
            else:
                hists.fill(
                    data[var],
                    sample,
                )

    print("------------------------------------------------------------")

    with open(f'{odir}/{ch}_{var}.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_1dhists(year, ch, odir, var):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
    """

    # luminosity
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    luminosity = luminosity / 1000.
    f.close()

    # load the hists
    hists = {}
    with open(f'{odir}/{ch}_{var}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make plots per channel
    fig, ax = plt.subplots(figsize=(8, 5))
    for sample in hists.axes[1]:
        hep.histplot(hists[{'samples': sample}],
                     ax=ax,
                     label=simplified_labels[sample],
                     color=color_by_sample[sample],
                     linewidth=3)
    ax.set_xlabel(f"{var}")
    ax.set_ylabel("Events")
    ax.set_title(f'{ch} channel for the signal')
    hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)
    ax.legend()
    plt.savefig(f'{odir}/1dhist_sig_{ch}_{var}.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for sample in hists.axes[1]:
        hep.histplot(hists[{'samples': sample}],
                     ax=ax,
                     label=simplified_labels[sample],
                     color=color_by_sample[sample],
                     linewidth=3)
    ax.set_xlabel(f"{var}")
    ax.set_ylabel("Events")
    ax.set_title(f'{ch} channel for the signal')
    ax.set_yscale("log")
    ax.set_ylim(0.1)

    ax.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left', title=f"{label_by_ch[ch]} Channel"
              )

    hep.cms.lumitext("%.1f " % luminosity + r"fb$^{-1}$ (13 TeV)", ax=ax, fontsize=20)
    hep.cms.text("Work in Progress", ax=ax, fontsize=15)
    plt.savefig(f'{odir}/1dhist_sig_{ch}_{var}_log.pdf')
    plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/1d_hists/'):
        os.makedirs(odir + '/1d_hists/')
    odir = odir + '/1d_hists/'

    channels = args.channels.split(',')
    range = [args.start, args.end]

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    samples[args.year] = {}
    for ch in channels:
        samples[args.year][ch] = []
        for key, value in json_samples[args.year][ch].items():
            if value == 1:
                samples[args.year][ch].append(key)

    for ch in channels:
        if args.make_hists:
            print(f'Making {args.var} histogram')
            make_1dhists(args.year, ch, args.idir, odir, samples, args.var, args.bins, range)

        if args.plot_hists:
            print(f'Plotting...')
            plot_1dhists(args.year, ch, odir, args.var)


if __name__ == "__main__":
    # e.g. run locally as
    # gen_Hpt_pt:   python make_1dhists_sig.py --year 2017 --odir sig --channels ele --var gen_Hpt    --bins 25 --start 300 --end 1000 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2_2017 --plot_hists --make_hists
    # fj_msoftdrop: python make_1dhists_sig.py --year 2017 --odir sig --channels ele --var fj_msoftdrop --bins 25 --start 25 --end 200 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2_2017 --plot_hists --make_hists
    # lep_fj_m:     python make_1dhists_sig.py --year 2017 --odir sig --channels ele --var lep_fj_m --bins 25 --start 25 --end 200 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2_2017 --plot_hists --make_hists

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json",
                        help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',
                        help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists',
                        help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',
                        help="input directory with results")
    parser.add_argument('--var',             dest='var',         default='lep_pt',
                        help='variable to plot')
    parser.add_argument('--bins',            dest='bins',        default=50,
                        help="binning of the first variable passed",                type=int)
    parser.add_argument('--start',           dest='start',       default=0,
                        help="starting range of the first variable passed",         type=int)
    parser.add_argument('--end',             dest='end',         default=1,
                        help="end range of the first variable passed",              type=int)
    parser.add_argument("--make_hists",      dest='make_hists',
                        action='store_true',                        help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',
                        action='store_true',                        help="Plot the hists")

    args = parser.parse_args()

    main(args)
