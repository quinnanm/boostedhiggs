#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018
from utils import get_simplified_label, get_sum_sumgenweight
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
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_2dplots(year, ch, idir, odir, samples, vars, x_bins, x_start, x_end, y_bins, y_start, y_end):
    """
    Makes 2D plots of two variables

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
        vars: a list of two variable names to plot against each other... see the full list of choices in plot_configs/vars.json
    """

    # instantiates the histogram object
    hists = hist2.Hist(
        hist2.axis.Regular(x_bins, x_start, x_end, name=vars[0], label=vars[0], overflow=True),
        hist2.axis.Regular(y_bins, y_start, y_end, name=vars[1], label=vars[1], overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),
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

        # get list of parquet files that have been processed
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

            single_sample = None
            for single_key, key in add_samples.items():
                if key in sample:
                    single_sample = single_key

            # combining all pt bins of a specefic process under one name
            if single_sample is not None:
                hists.fill(
                    data[vars[0]],
                    data[vars[1]],
                    single_sample,
                )
            # otherwise give unique name
            else:
                hists.fill(
                    data[vars[0]],
                    data[vars[1]],
                    sample,
                )

    print("------------------------------------------------------------")

    with open(f'{odir}/{ch}_{vars[0]}_{vars[1]}.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_2dplots(year, ch, odir, vars):
    """
    Plots 2D plots of two variables that were made by "make_2dplots" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        vars: a list of two variable names to plot against each other... see the full list of choices in plot_configs/vars.json
    """

    # load the hists
    with open(f'{odir}/{ch}_{vars[0]}_{vars[1]}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make directory to store stuff per year
    if not os.path.exists(f'{odir}/{ch}_{vars[0]}_{vars[1]}'):
        os.makedirs(f'{odir}/{ch}_{vars[0]}_{vars[1]}')

    # make plots per channel
    for sample in hists.axes[2]:
        # one for log z-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        hep.hist2dplot(hists[{'samples': sample}], ax=ax, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1000))
        ax.set_xlabel(f"{vars[0]}")
        ax.set_ylabel(f"{vars[1]}")
        ax.set_title(f'{ch} channel for \n {sample}')
        hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        plt.savefig(f'{odir}/{ch}_{vars[0]}_{vars[1]}/{sample}_log_z.pdf')
        plt.close()

        # one for non-log z-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        hep.hist2dplot(hists[{'samples': sample}], ax=ax, cmap="plasma")
        ax.set_xlabel(f"{vars[0]}")
        ax.set_ylabel(f"{vars[1]}")
        ax.set_title(f'{ch} channel for \n {sample}')
        hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        plt.savefig(f'{odir}/{ch}_{vars[0]}_{vars[1]}/{sample}.pdf')
        plt.close()


def main(args):

    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/2d_plots/'):
        os.makedirs(odir + '/2d_plots/')
    odir = odir + '/2d_plots/'

    channels = args.channels.split(',')

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
    vars = args.vars.split(',')

    for ch in channels:

        if args.make_hists:
            print(f'Making 2dplot of {vars} for {ch} channel')
            make_2dplots(args.year, ch, args.idir, odir, samples, vars, args.x_bins, args.x_start, args.x_end, args.y_bins, args.y_start, args.y_end)

        if args.plot_hists:
            print('Plotting...')
            plot_2dplots(args.year, ch, odir, vars)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt vs lep_iso:   python make_2dplots.py --year 2017 --odir hists --channels ele --vars lep_pt,lep_isolation --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 1 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs lep_fj_dr: python make_2dplots.py --year 2017 --odir hists --channels ele --vars lep_pt,lep_fj_dr     --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0.1 --y_end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # fj_pt vs lep_fj_dr:  python make_2dplots.py --year 2017 --odir hists --channels ele,mu --vars fj_pt,lep_fj_dr      --make_hists --plot_hists --x_bins 100 --x_start 200 --x_end 500 --y_bins 100 --y_start 0.1 --y_end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs mt:        python make_2dplots.py --year 2017 --odir hists --channels ele --vars lep_pt,lep_met_mt    --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs fj_pt:     python make_2dplots.py --year 2017 --odir hists --channels ele --vars lep_pt,fj_pt         --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                                 help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json",     help="path to json with samples to be plotted")
    parser.add_argument('--channels',        dest='channels',    default='ele',                                  help="channel... choices are ['ele', 'mu', 'had']")
    parser.add_argument('--odir',            dest='odir',        default='hists',                                help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                          help="input directory with results")
    parser.add_argument('--vars',            dest='vars',        default='lep_pt,lep_isolation',                 help="channels for which to plot this variable")
    parser.add_argument('--x_bins',          dest='x_bins',      default=50,                                     help="binning of the first variable passed",                type=int)
    parser.add_argument('--x_start',         dest='x_start',     default=0,                                      help="starting range of the first variable passed",         type=float)
    parser.add_argument('--x_end',           dest='x_end',       default=1,                                      help="end range of the first variable passed",              type=float)
    parser.add_argument('--y_bins',          dest='y_bins',      default=50,                                     help="binning of the second variable passed",               type=int)
    parser.add_argument('--y_start',         dest='y_start',     default=0,                                      help="starting range of the second variable passed",        type=float)
    parser.add_argument('--y_end',           dest='y_end',       default=1,                                      help="end range of the second variable passed",             type=float)
    parser.add_argument("--make_hists",      dest='make_hists',  action='store_true',                            help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',  action='store_true',                            help="Plot the hists")

    args = parser.parse_args()

    main(args)
