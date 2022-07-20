#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, label_by_ch
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

    max_iso = {'ele': 120, 'mu': 55}

    # isolation
    hists_iso = hist2.Hist(
        hist2.axis.Regular(30, 0, 4, name='lep_isolation', label='lep_isolation', overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),
    )

    hists_miso = hist2.Hist(
        hist2.axis.Regular(30, 0, 4, name='lep_misolation', label='lep_misolation', overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),
    )

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

            single_sample = None
            for single_key, key in add_samples.items():
                if key in sample:
                    single_sample = single_key

            select_iso = data['lep_pt'] < max_iso[ch]
            select_miso = data['lep_pt'] > max_iso[ch]

            if single_sample is not None:
                hists_iso.fill(
                    data['lep_isolation'][select_iso],
                    single_sample,  # combining all events under one name
                    weight=event_weight[select_iso],
                )
                selection = data['lep_pt'] > max_iso[ch]
                hists_miso.fill(
                    data['lep_misolation'][select_miso],
                    single_sample,  # combining all events under one name
                    weight=event_weight[select_miso],
                )
            else:
                hists_iso.fill(
                    data['lep_isolation'][select_iso],
                    sample,
                    weight=event_weight[select_iso],
                )
                hists_miso.fill(
                    data['lep_misolation'][select_miso],
                    sample,
                    weight=event_weight[select_miso],
                )

    print("------------------------------------------------------------")

    with open(f'{odir}/cut_{ch}_iso.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists_iso, f)
    with open(f'{odir}/cut_{ch}_miso.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists_miso, f)


def plot_stacked_hists(year, ch, odir, cut):
    """
    Plots the stacked 1D histograms that were made by "make_stacked_hists" individually for each year
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        vars_to_plot: the set of variable to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
    """

    # load the hists
    if cut == 'iso':
        with open(f'{odir}/cut_{ch}_iso.pkl', 'rb') as f:
            hists_iso = pkl.load(f)
            f.close()
    elif cut == 'miso':
        with open(f'{odir}/cut_{ch}_miso.pkl', 'rb') as f:
            hists_iso = pkl.load(f)
            f.close()

    # make the histogram plots in this directory
    if not os.path.exists(f'{odir}/cut_plots'):
        os.makedirs(f'{odir}/cut_plots')

    data_label = data_by_ch[ch]

    # get samples existing in histogram
    samples = [hists_iso.axes[1].value(i) for i in range(len(hists_iso.axes[1].edges))]
    signal_labels = [label for label in samples if label in signal_by_ch[ch]]
    bkg_labels = [label for label in samples if (label and label != data_label and label not in signal_labels)]

    # signal
    signal = [hists_iso[{"samples": label}] for label in signal_labels]

    # background
    bkg = [hists_iso[{"samples": label}] for label in bkg_labels]

    fig, ax = plt.subplots(1, 1)

    errps = {
        'hatch': '////',
        'facecolor': 'none',
        'lw': 0,
        'color': 'k',
        'edgecolor': (0, 0, 0, .5),
        'linewidth': 0,
        'alpha': 0.4
    }
    if len(bkg) > 0:
        hep.histplot(bkg,
                     ax=ax,
                     stack=True,
                     sort='yield',
                     histtype="fill",
                     label=[get_simplified_label(bkg_label) for bkg_label in bkg_labels],
                     )
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handle.set_color(color_by_sample[label])

        tot = bkg[0].copy()
        for i, b in enumerate(bkg):
            if i > 0:
                tot = tot + b
        ax.stairs(
            values=tot.values() + np.sqrt(tot.values()),
            baseline=tot.values() - np.sqrt(tot.values()),
            edges=tot.axes[0].edges, **errps,
            label='Stat. unc.'
        )

    if len(signal) > 0:
        hep.histplot(signal,
                     ax=ax,
                     label=[get_simplified_label(sig_label) for sig_label in signal_labels],
                     color='red'
                     )
        sig = signal[0].copy()
        for i, s in enumerate(signal):
            if i > 0:
                sig = sig + s
        ax.stairs(
            values=sig.values() + np.sqrt(sig.values()),
            baseline=sig.values() - np.sqrt(sig.values()),
            edges=sig.axes[0].edges, **errps,
        )

    ax.set_yscale('log')
    ax.set_ylim(0.1)
    ax.set_title(f'{label_by_ch[ch]} Channel')
    ax.legend()

    hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)

    print(f'Saving to {odir}/cut_plots/hists_{cut}.pdf')
    plt.savefig(f'{odir}/cut_plots/hists_{cut}.pdf', bbox_inches='tight')
    plt.close()

#
# def plot_1dhists(year, ch, odir):
#     """
#     Plots 1D histograms that were made by "make_1dhists" function
#
#     Args:
#         year: string that represents the year the processed samples are from
#         ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
#         odir: output directory to hold the plots
#     """
#
#     # load the hists
#     with open(f'{odir}/cut_{ch}_iso.pkl', 'rb') as f:
#         hists_iso = pkl.load(f)
#         f.close()
#
#     with open(f'{odir}/cut_{ch}_miso.pkl', 'rb') as f:
#         hists_miso = pkl.load(f)
#         f.close()
#
#     # make directory to store stuff per year
#     if not os.path.exists(f'{odir}/cut_plots'):
#         os.makedirs(f'{odir}/cut_plots')
#
#     # make plots per channel
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for sample in hists_iso.axes[1]:
#         hep.histplot(hists_iso[{'samples': sample}], ax=ax, label=sample)
#     hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
#     hep.cms.text("Work in Progress", ax=ax)
#     ax.legend()
#     ax.set_yscale('log')
#     ax.set_ylabel('Events')
#     ax.set_title(f'Lepton isolation \n for {ch} channel', fontsize=14)
#     hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
#     hep.cms.text("Work in Progress", ax=ax)
#     plt.savefig(f'{odir}/cut_plots/{ch}_lep_iso.pdf')
#     plt.close()
#
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for sample in hists_miso.axes[1]:
#         hep.histplot(hists_miso[{'samples': sample}], ax=ax, label=sample)
#     hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
#     hep.cms.text("Work in Progress", ax=ax)
#     ax.legend()
#     ax.set_yscale('log')
#     ax.set_ylabel('Events')
#     ax.set_title(f'Lepton misolation \n for {ch} channel', fontsize=14)
#     hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
#     hep.cms.text("Work in Progress", ax=ax)
#     plt.savefig(f'{odir}/cut_plots/{ch}_lep_miso.pdf')
#     plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/1d_cuts/'):
        os.makedirs(odir + '/1d_cuts/')
    odir = odir + '/1d_cuts/'

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
            # plot_1dhists(args.year, ch, odir)
            for cut in ['iso', 'miso']:
                print(f'Plotting for {cut} cut...')
                plot_stacked_hists(args.year, ch, odir, cut)


if __name__ == "__main__":
    # e.g. run locally as
    # python cut_histograms.py --year 2017 --odir plots --channels ele --plot_hists --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/ --make_hists

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
