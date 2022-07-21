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


def count_s_over_b(year, channels, idir, odir, samples, cut):
    """
    Counts signal and background at different working points of a cut

    Args:
        year: string that represents the year the processed samples are from
        channels: list of channels... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    for ch in channels:
        c = 0
        # loop over the samples
        for sample in samples[year][ch]:
            print(f"{sample}")

            # skip data samples
            is_data = False
            for key in data_by_ch.values():
                if key in sample:
                    is_data = True
            if is_data:
                continue

            # check if the sample was processed
            pkl_dir = f'{idir}/{sample}/outfiles/*.pkl'
            pkl_files = glob.glob(pkl_dir)  #
            if not pkl_files:  # skip samples which were not processed
                continue

            # check if the sample was processed
            parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')

            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    continue

                try:
                    event_weight = data['tot_weight']
                except:
                    continue

                single_sample = None
                for single_key, key in add_samples.items():
                    if key in sample:
                        single_sample = single_key

                if single_sample is not None:
                    if c == 0:
                        data1 = pd.DataFrame(data['lep_isolation'])
                        data1['sample'] = single_sample
                        c = c + 1
                    else:
                        data2 = pd.DataFrame(data['lep_isolation'])
                        data2['sample'] = single_sample

                        data1 = pd.concat([data1, data2])
                else:
                    if c == 0:
                        data1 = pd.DataFrame(data['lep_isolation'])
                        data1['sample'] = sample
                        c = c + 1
                    else:
                        data2 = pd.DataFrame(data['lep_isolation'])
                        data2['sample'] = sample

                        data1 = pd.concat([data1, data2])

            print("------------------------------------------------------------")

        with open(f'{odir}/data1.pkl', 'wb') as f:  # saves the hists objects
            pkl.dump(data1, f)


def plot_s_over_b(year, channels, odir, cut):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    with open(f'{odir}/wp_{cut}.pkl', 'rb') as f:
        wp = pkl.load(f)
        f.close()
    with open(f'{odir}/counts_{cut}.pkl', 'rb') as f:
        counts = pkl.load(f)
        f.close()
    with open(f'{odir}/counts_before_{cut}.pkl', 'rb') as f:
        s_over_b_before_cut = pkl.load(f)
        f.close()

    # s/b for b=DY,TTbar,Wjets
    fig, ax = plt.subplots(figsize=(8, 8))

    for ch in channels:
        num = counts[ch]['GluGluHToWWToLNuQQ']
        deno = [sum(x) for x in zip(counts[ch]['DYJets'], counts[ch]['TTbar'], counts[ch]['WJetsLNu'])]

        legend = s_over_b_before_cut[ch]['GluGluHToWWToLNuQQ'] / np.sqrt((s_over_b_before_cut[ch]['DYJets'] + s_over_b_before_cut[ch]['TTbar'] + s_over_b_before_cut[ch]['WJetsLNu']))

        ax.plot(wp[ch], num / np.sqrt(deno), label=f'{ch} channel, with s/b before cut = {str(round(legend,3))}')

    # ax.set_yscale('log')
    if cut == 'met_lep':
        ax.set_title('s/$\sqrt{b}$ as a function of the met_pt/lep_pt cut \n with DY, TTbar, Wjets background', fontsize=16)
        ax.set_xlabel(r'$\frac{pT_{met}}{pT_{lep}}$<x', fontsize=15)
    elif cut == 'dphi':
        ax.set_title('s/$\sqrt{b}$ as a function of the dphi cut \n with DY, TTbar, Wjets background', fontsize=16)
        ax.set_xlabel('|dphi(met, jet)<x|', fontsize=15)
    elif cut == 'iso':
        ax.set_title('s/$\sqrt{b}$ as a function of the lepton isolation cut \n with DY, TTbar, Wjets background', fontsize=16)
        ax.set_xlabel('lep_iso<x', fontsize=15)
    elif cut == 'miso':
        ax.set_title('s/$\sqrt{b}$ as a function of the lepton mini-isolation cut \n with DY, TTbar, Wjets background', fontsize=16)
        ax.set_xlabel('lep_miso<x', fontsize=15)

    ax.set_ylabel(r's/$\sqrt{b}$', fontsize=15)
    ax.legend()
    plt.savefig(f'{odir}/{cut}_s_over_b_dy_tt_wjets.pdf')
    plt.close()

    # s/b for b=QCD
    fig, ax = plt.subplots(figsize=(8, 8))
    for ch in channels:
        num = counts[ch]['GluGluHToWWToLNuQQ']
        deno = counts[ch]['QCD']
        legend = s_over_b_before_cut[ch]['GluGluHToWWToLNuQQ'] / np.sqrt(s_over_b_before_cut[ch]['QCD'])

        ax.plot(wp[ch], num / np.sqrt(deno), label=f'{ch} channel, with s/b before cut = {str(round(legend,3))}')

    # ax.set_yscale('log')
    if cut == 'met_lep':
        ax.set_title('s/$\sqrt{b}$ as a function of the met_pt/lep_pt cut \n with QCD background', fontsize=16)
        ax.set_xlabel(r'$\frac{pT_{met}}{pT_{lep}}$<x', fontsize=15)
    elif cut == 'dphi':
        ax.set_title('s/$\sqrt{b}$ as a function of the dphi cut \n with QCD background', fontsize=16)
        ax.set_xlabel('|dphi(met, jet)<x|', fontsize=15)
    elif cut == 'iso':
        ax.set_title('s/$\sqrt{b}$ as a function of the lepton isolation cut \n with QCD background', fontsize=16)
        ax.set_xlabel('lep_iso<x', fontsize=15)
    elif cut == 'miso':
        ax.set_title('s/$\sqrt{b}$ as a function of the lepton mini-isolation cut \n with QCD background', fontsize=16)
        ax.set_xlabel('lep_miso<x', fontsize=15)

    ax.set_ylabel(r's/$\sqrt{b}$', fontsize=15)
    ax.legend()
    plt.savefig(f'{odir}/{cut}_s_over_b_qcd.pdf')
    plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/s_over_b/'):
        os.makedirs(odir + '/s_over_b/')
    odir = odir + '/s_over_b'

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

    if args.make_counts:
        for cut in ['iso', 'miso', 'dphi']:
            # for cut in ['iso']:
            print(f'counting s/b after {cut} cut')
            count_s_over_b(args.year, channels, args.idir, odir, samples, cut)

    if args.plot_counts:
        for cut in ['iso', 'miso', 'dphi']:
            # for cut in ['iso']:
            print(f'plotting s/b for {cut} cut')
            plot_s_over_b(args.year, channels, odir, cut)


if __name__ == "__main__":
    # e.g. run locally as
    # python lep_iso.py --year 2017 --odir plots --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/ --plot_counts --make_counts

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
