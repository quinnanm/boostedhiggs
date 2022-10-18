#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018
from utils import get_sample_to_use, get_simplified_label, get_sum_sumgenweight
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


def make_1dhists(year, ch, idir, odir, samples, bins, range):
    """
    Makes 1D histograms

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """
    # Define cuts to make later
    pt_iso = {"ele": 120, "mu": 55}

    # instantiates the histogram object
    hists = hist2.Hist(
        hist2.axis.StrCategory([], name='samples', growth=True),     # to combine different pt bins of the same process
        hist2.axis.Regular(bins, range[0], range[1], name='ch_weight', label='channel weight', overflow=True),
        hist2.axis.Regular(bins, range[0], range[1], name='plain_weight', label='plain weight', overflow=True),
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

        # get combined sample
        sample_to_use = get_sample_to_use(sample, year)

        for i, parquet_file in enumerate(parquet_files):
            try:
                data = pq.read_table(parquet_file).to_pandas()
            except:
                print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                continue

            if len(data) == 0:
                print("Parquet file empty")
                continue

            try:
                event_weight = data["tot_weight"]
            except:
                print("No tot_weight variable in parquet - run pre-processing first!")
                continue

            if ch == "mu":
                data['mu_score'] = data['fj_isHVV_munuqq'] / \
                    (data['fj_isHVV_munuqq'] + data['fj_ttbar_bmerged'] +
                        data['fj_ttbar_bsplit'] + data['fj_wjets_label'])
            elif ch == "ele":
                data['ele_score'] = data['fj_isHVV_elenuqq'] / \
                    (data['fj_isHVV_elenuqq'] + data['fj_ttbar_bmerged'] +
                        data['fj_ttbar_bsplit'] + data['fj_wjets_label'])

            # make kinematic cuts
            pt_cut = (data["fj_pt"] > 400) & (data["fj_pt"] < 600)
            msd_cut = (data["fj_msoftdrop"] > 30) & (data["fj_msoftdrop"] < 150)

            # make isolation cuts
            iso_cut = (
                ((data["lep_isolation"] < 0.15) & (data["lep_pt"] < pt_iso[ch])) |
                (data["lep_pt"] > pt_iso[ch])
            )

            # make mini-isolation cuts
            if ch == "mu":
                miso_cut = (
                    ((data["lep_misolation"] < 0.1) & (data["lep_pt"] >= pt_iso[ch])) |
                    (data["lep_pt"] < pt_iso[ch])
                )
            elif ch == "ele":
                miso_cut = data["lep_pt"] > 10
                
            select_var = iso_cut & miso_cut
                    
            # filling histograms
            hists.fill(
                samples=sample_to_use,
                ch_weight=data[f"weight_{ch}"][select_var],
                plain_weight=data[f"weight"][select_var],
            )

    print("------------------------------------------------------------")

    with open(f'{odir}/{ch}_event_weight.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_1dhists(year, channels, odir):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    for ch in channels:
        with open(f'{odir}/{ch}_event_weight.pkl', 'rb') as f:
            h = pkl.load(f)
            f.close()

        # make plots per channel
        for sample in h.axes[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            hep.histplot(h[{'samples': sample, 'ch_weight': sum}], ax=ax, label='plain_weight')
            hep.histplot(h[{'samples': sample, 'plain_weight': sum}], ax=ax, label='ch_weight')
            ax.set_xlabel(f"event weight")
            ax.set_title(f'{ch} channel for \n {sample}')
            hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
            hep.cms.text("Work in Progress", ax=ax)
            ax.legend()
            plt.savefig(f'{odir}/{ch}_event_weight_{sample}.pdf')
            plt.savefig(f'{odir}/{ch}_event_weight_{sample}.png')
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
            print(f'Making histogram')
            make_1dhists(args.year, ch, args.idir, odir, samples, args.bins, range)

    if args.plot_hists:
        print(f'Plotting...')
        plot_1dhists(args.year, channels, odir)


if __name__ == "__main__":
    # e.g. run locally as
    # python make_1dhists_event_weight.py --year 2017 --odir hists --channels ele,mu --bins 100 --start 0 --end 1 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2_2017 --plot_hists --make_hists
    
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
