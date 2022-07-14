#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
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


def count_s_over_b(year, channels, idir, odir, samples):
    """
    Counts signal and background at different working points of a cut

    Args:
        year: string that represents the year the processed samples are from
        channels: list of channels... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    wp, count_sig, count_bkg = {}, {}, {}

    for ch in channels:
        wp[ch], count_sig[ch], count_bkg[ch] = [], [], []

        c_sig, c_bkg = 0, 0
        # for i in range(0, 400, 4):
        for i in range(0, 100, 2):
            print(f'Processing working point {i * 0.01}')
            wp[ch].append(i * 0.01)      # working point

            # loop over the samples
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
                    print(f'- Processing {ch} channel of sample', sample)

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        if is_data:
                            print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        else:
                            print('Not able to read data from ', parquet_file)
                        continue

                    try:
                        event_weight = data['tot_weight']
                    except:
                        print('No tot_weight variable in parquet - run pre-processing first!')
                        continue

                    if sample == 'GluGluHToWWToLNuQQ':
                        print('signal')
                        c_sig = c_sig + (data['tot_weight'] * ((abs(data['lep_isolation'])) < (i * 0.01))).sum()
                    else:
                        print('background')
                        c_bkg = c_bkg + (data['tot_weight'] * ((abs(data['lep_isolation'])) < (i * 0.01))).sum()

            count_sig[ch].append(c_sig)   # cut defined at the working point
            count_bkg[ch].append(c_bkg)   # cut defined at the working point

        print("------------------------------------------------------------")

        with open(f'{odir}/wp.pkl', 'wb') as f:  # saves the hists objects
            pkl.dump(wp, f)
        with open(f'{odir}/count_sig.pkl', 'wb') as f:  # saves the hists objects
            pkl.dump(count_sig, f)
        with open(f'{odir}/count_bkg.pkl', 'wb') as f:  # saves the hists objects
            pkl.dump(count_bkg, f)


def plot_s_over_b(year, channels, odir):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
    """

    # load the hists
    with open(f'{odir}/wp.pkl', 'rb') as f:
        wp = pkl.load(f)
        f.close()
    with open(f'{odir}/count_sig.pkl', 'rb') as f:
        count_sig = pkl.load(f)
        f.close()
    with open(f'{odir}/count_bkg.pkl', 'rb') as f:
        count_bkg = pkl.load(f)
        f.close()

    fig, ax = plt.subplots()

    for ch in channels:
        ax.plot(wp[ch], count_sig[ch] / np.sqrt(count_bkg[ch]), label=f'{ch} channel')
    # ax.set_yscale('log')
    ax.set_title(r's/$\sqrt{b}$ as a function of the dphi cut', fontsize=16)
    ax.set_ylabel(r's/$\sqrt{b}$', fontsize=15)
    ax.set_xlabel('|dphi(met, jet)<x|', fontsize=15)
    ax.legend()
    plt.savefig(f'{odir}/s_over_b.pdf')


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/s_over_b/'):
        os.makedirs(odir + '/s_over_b/')
    odir = odir + '/s_over_b'

    if not os.path.exists(odir + '/' + args.tag):
        os.makedirs(odir + '/' + args.tag)
    odir = odir + '/' + args.tag

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

    if args.make_counts:
        print(f'counting s/b')
        count_s_over_b(args.year, channels, args.idir, odir, samples)

    if args.plot_counts:
        print(f'plotting s/b')
    plot_s_over_b(args.year, channels, odir)


if __name__ == "__main__":
    # e.g. run locally as
    # python s_over_b.py --year 2017 --odir plots --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/ --tag iso --make_counts --plot_counts

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists',                            help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument('--tag',             dest='tag',         default='',                           help='str to append for saving the count variables')
    parser.add_argument("--make_counts",      dest='make_counts',  action='store_true',                        help="Make hists")
    parser.add_argument("--plot_counts",      dest='plot_counts',  action='store_true',                        help="Plot the hists")

    args = parser.parse_args()

    main(args)
