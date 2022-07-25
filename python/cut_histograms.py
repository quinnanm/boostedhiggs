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


def make_1dhists(year, ch, idir, odir, samples, cuts):
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

    hists = {}
    # isolation
    if 'iso' in cuts:
        hists['iso'] = hist2.Hist(
            hist2.axis.Regular(30, 0, 2, name='lep_isolation', label='lep_isolation', overflow=True),
            hist2.axis.StrCategory([], name='samples', growth=True),
        )
    if 'miso' in cuts:
        hists['miso'] = hist2.Hist(
            hist2.axis.Regular(30, 0, 2, name='lep_misolation', label='lep_misolation', overflow=True),
            hist2.axis.StrCategory([], name='samples', growth=True),
        )
    if 'dphi' in cuts:
        hists['dphi'] = hist2.Hist(
            hist2.axis.Regular(30, 1, 3.14, name='dphi', label='dphi', overflow=True),
            hist2.axis.StrCategory([], name='samples', growth=True),
        )
    if 'met_lep' in cuts:
        hists['met_lep'] = hist2.Hist(
            hist2.axis.Regular(30, 1, 2, name='met_lep', label='met_lep', overflow=True),
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
            if single_sample is not None:
                sample_to_use = single_sample   # combining all events under one name
            else:
                sample_to_use = sample

            if 'iso' in cuts:
                select_iso = data['lep_pt'] < max_iso[ch]
                hists['iso'].fill(
                    data['lep_isolation'][select_iso],
                    sample_to_use,
                    weight=event_weight[select_iso],
                )
            if 'miso' in cuts:
                select_miso = data['lep_pt'] > max_iso[ch]
                hists['miso'].fill(
                    data['lep_misolation'][select_miso],
                    sample_to_use,
                    weight=event_weight[select_miso],
                )
            if 'dphi' in cuts:
                select = abs(data['met_fj_dphi']) < 1
                hists['dphi'].fill(
                    abs(data['met_fj_dphi'])[select],
                    sample_to_use,
                    weight=event_weight[select],
                )
            if 'met_lep' in cuts:
                select = (data['met'] / data['lep_pt']) < 1
                hists['met_lep'].fill(
                    (data['met'] / data['lep_pt'])[select],
                    sample_to_use,
                    weight=event_weight[select],
                )

    print("------------------------------------------------------------")

    for cut in cuts:
        with open(f'{odir}/cut_{ch}_{cut}.pkl', 'wb') as f:  # saves the hists objects
            pkl.dump(hists[cut], f)


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
    with open(f'{odir}/cut_{ch}_{cut}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make the histogram plots in this directory
    if not os.path.exists(f'{odir}/cut_plots'):
        os.makedirs(f'{odir}/cut_plots')

    data_label = data_by_ch[ch]

    # get samples existing in histogram
    samples = [hists.axes[1].value(i) for i in range(len(hists.axes[1].edges))]
    signal_labels = [label for label in samples if label in signal_by_ch[ch]]
    bkg_labels = [label for label in samples if (label and label != data_label and label not in signal_labels)]

    # signal
    signal = [hists[{"samples": label}] for label in signal_labels]

    # background
    bkg = [hists[{"samples": label}] for label in bkg_labels]

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
    if cut == 'met_lep':
        ax.set_xlabel(r'$\frac{pT_{met}}{pT_{lep}}$', fontsize=15)

    hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)

    print(f'Saving to {odir}/cut_plots/{ch}_hists_{cut}_after_cut.pdf')
    plt.savefig(f'{odir}/cut_plots/{ch}_hists_{cut}_after_cut.pdf', bbox_inches='tight')
    plt.close()


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

    # cuts = ['iso', 'miso', 'dphi']
    cuts = ['dphi', 'met_lep']

    for ch in channels:
        if args.make_hists:
            print(f'Making iso and miso cut histograms')
            make_1dhists(args.year, ch, args.idir, odir, samples, cuts)

        if args.plot_hists:
            for cut in cuts:
                print(f'Plotting for {cut} cut...')
                plot_stacked_hists(args.year, ch, odir, cut)


if __name__ == "__main__":
    # e.g. run locally as
    # python cut_histograms.py --year 2017 --odir plots --channels ele,mu --plot_hists --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/ --make_hists

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
