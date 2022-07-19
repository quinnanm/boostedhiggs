#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
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


def make_1dhists(year, ch, idir, odir, samples):
    """
    Makes 1D histograms

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    # for each cut/working point we will have a lep_pt distribution of the events passing the cut

    max_iso = {'ele': 120, 'mu': 55}

    hists = hist2.Hist(
        hist2.axis.Regular(20, 0, 1, name='lep_miso<x', label='lep_miso<x', overflow=True),
        hist2.axis.Regular(20, 30, 320, name='lep_pt', label='lep_pt', overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),
    )

    for sample in samples[year][ch]:
        # skip data samples
        is_data = False
        for key in data_by_ch.values():
            if key in sample:
                is_data = True
        if is_data:
            continue

        print("------------------------------------------------------------")

        # check if the sample was processed
        pkl_dir = f'{idir}/{sample}/outfiles/*.pkl'
        pkl_files = glob.glob(pkl_dir)  #
        if not pkl_files:  # skip samples which were not processed
            print('- No processed files found...', pkl_dir, 'skipping sample...', sample)
            continue

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
                event_weight = data['tot_weight']
            except:
                print('No tot_weight variable in parquet - run pre-processing first!')
                continue

        wp = []
        y_lep = []
        for i in range(0, 400, 1):
            wp.append(i * 0.01)      # working point
            cut = (data['lep_misolation'] < (i * 0.01)) & (data['lep_pt'] > max_iso[ch])   # cut defined at the working point
            y_lep.append(data['lep_pt'][cut].to_numpy())  # lepton_pt distribution of events passing the cut

        # to plot the 2d map -> (1) expand the wp list (2) unfold the y_lep sublists
        # recall: wp is a list of working points
        # recall: y_lep is a list of distributions per working point
        x = []
        y = []
        for i in range(len(wp)):
            for j in y_lep[i]:
                x.append(wp[i])
                y.append(j)

        single_sample = None
        for single_key, key in add_samples.items():
            if key in sample:
                single_sample = single_key

        if single_sample is not None:
            hists.fill(
                x,
                y,
                single_sample,
            )
        else:
            hists.fill(
                x,
                y,
                sample,
            )

    print("------------------------------------------------------------")

    with open(f'{odir}/2d_cut_{ch}_miso.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_1dhists(year, ch, odir):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    with open(f'{odir}/2d_cut_{ch}_miso.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make directory to store stuff per year
    if not os.path.exists(f'{odir}/2d_cut_plots'):
        os.makedirs(f'{odir}/2d_cut_plots')

    # make plots per channel
    for sample in hists.axes[1]:
        fig, ax = plt.subplots(figsize=(8, 5))
        hep.hist2dplot(hists[{'samples': sample}], ax=ax, cmap="plasma")
        ax.set_title(f'Lepton pT distribution for {sample_mapping[sample]} \n for the {ch} channel at different working points', fontsize=16)
        ax.set_ylabel('lep_pt', fontsize=15)
        ax.set_xlabel('lep_miso<x', fontsize=15)
        plt.savefig(f'plots/2d_cut_plots/2d_miso_{ch}_{sample}.pdf')


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
            print(f'Making iso and miso cut histograms')
            make_1dhists(args.year, ch, args.idir, odir, samples)

        if args.plot_hists:
            print(f'Plotting...')
            plot_1dhists(args.year, ch, odir)


if __name__ == "__main__":
    # e.g. run locally as
    # python cut_2d.py --year 2017 --odir cuts --channels ele --make_hists --plot_hists --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists',                            help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument("--make_hists",      dest='make_hists',  action='store_true',                        help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',  action='store_true',                        help="Plot the hists")

    args = parser.parse_args()

    main(args)
