#!/usr/bin/python

from BoolArg import BoolArg
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")

# define the axes for the different variables to be plotted
axis_dict = {
    'lepton_pt': hist2.axis.Regular(50, 20, 500, name='var', label=r'Lepton $p_T$ [GeV]'),
    'lep_isolation': hist2.axis.Regular(20, 0, 3.5, name='var', label=r'Lepton iso'),
    'ht':  hist2.axis.Regular(20, 180, 1500, name='var', label='HT [GeV]'),
    'dr_jet_candlep': hist2.axis.Regular(20, 0, 1.5, name='var', label=r'$\Delta R(l, Jet)$'),
    'met':  hist2.axis.Regular(50, 0, 400, name='var', label='MET [GeV]'),
    'mt_lep_met':  hist2.axis.Regular(20, 0, 300, name='var', label=r'$m_T(lep, p_T^{miss})$ [GeV]'),
    'mu_mvaId': hist2.axis.Regular(20, -1, 1, name='var', label='Muon MVAID'),
    'leadingfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet $p_T$ [GeV]'),
    'leadingfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet $m_{sd}$ [GeV]'),
    'secondfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'2nd Jet $p_T$ [GeV]'),
    'secondfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'2nd Jet $m_{sd}$ [GeV]'),
    'bjets_ophem_leadingfj': hist2.axis.Regular(20, 0, 1, name='var', label=r'btagFlavB (opphem)'),
}


def get_simplified_label(sample, pfnano):   # get simplified "alias" names of the samples for plotting purposes
    if pfnano:
        return sample
    f = open('../data/simplified_labels.json')
    name = json.load(f)
    f.close()
    return str(name[sample])


def get_sum_sumgenweight(idir, year, sample):
    pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get the pkl metadata of the pkl files that were processed
    sum_sumgenweight = 0

    for file in pkl_files:
        # load and sum the sumgenweight of each
        with open(file, 'rb') as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]['sumgenweight']
    return sum_sumgenweight


def make_hist(idir, odir, vars_to_plot, samples, years, channels, pfnano):  # makes histograms and saves in pkl file
    hists = {}  # define a placeholder for all histograms
    for year in years:
        # Get luminosity of year
        f = open('../data/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}

        for ch in channels:  # initialize the histograms for the different channels and different variables
            hists[year][ch] = {}

            for var in vars_to_plot:
                sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)

                hists[year][ch][var] = hist2.Hist(
                    sample_axis,
                    axis_dict[var],
                )

        # loop over the processed files and fill the histograms
        for sample in samples:
            print('Processing sample', sample)
            pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get list of files that were processed
            if not pkl_files:  # skip samples which were not processed
                print('- No processed files found... skipping sample...')
                continue

            # Get xsection of sample
            if args.pfnano:
                f = open('../data/xsecs_pfnano.json')
            else:
                f = open('../data/xsecs.json')

            xsec = json.load(f)
            f.close()
            xsec = eval(str((xsec[sample])))  # because some xsections are given as string formulas in the xsecs.json
            print('- xsection of sample is', xsec)

            # Get sum_sumgenweight of sample
            sum_sumgenweight = get_sum_sumgenweight(idir, year, sample)

            # Get overall weighting of events
            xsec_weight = (xsec * luminosity[year]) / (sum_sumgenweight)  # each event has (possibly a different) genweight... sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks

            for ch in channels:
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed

                for parquet_file in parquet_files:
                    data = pq.read_table(parquet_file).to_pandas()

                    for var in vars_to_plot:
                        if var not in data.keys():
                            print(f'- No {var} for {year}/{ch} - skipping')
                            continue

                        # we can make further selections before filling the hists here
                        data = data[data['ht'] > 300]

                        variable = data[var].to_numpy()
                        try:
                            event_weight = data['weight'].to_numpy()
                        except:
                            event_weight = 1  # for data

                        # filling histograms
                        if "QCD" in sample:
                            hists[year][ch][var].fill(
                                samples="QCD",  # combining all QCD events under one name "QCD"
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )
                        elif "WJetsToLNu" in sample:  # combining all WJetsToLNu events under one name "WJetsToLNu"
                            hists[year][ch][var].fill(
                                samples="WJetsToLNu",
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )
                        else:
                            hists[year][ch][var].fill(
                                samples=get_simplified_label(sample, pfnano),
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )
            try:
                if event_weight == 1:
                    print(sample, "sample is data not MC")
            except:
                continue

    # store the hists variable
    with open(f'{odir}/hists.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def make_stack(odir, vars_to_plot, years, channels,pfnano,logy=True,add_data=False):
    if pfnano:
         signal_by_ch = {'ele': 'GluGluHToWWToLNuQQ',
                         'mu': 'GluGluHToWWToLNuQQ',
                         #'had': 'GluGluHToWWTo4q',
                         'had': 'GluGluHToWWToLNuQQ', #TODO: change this file
                    }
     else:
         signal_by_ch = {'ele': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                         'mu': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                         'had': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',  
                     }


    # load the hists
    with open(f'{odir}/hists.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make the histogram plots in this directory
    # TODO: we will want combined plots for all years later too
    for year in years:
        if not os.path.exists(f'{odir}/hists_{year}'):
            os.makedirs(f'{odir}/hists_{year}')
        for ch in channels:
            for var in vars_to_plot:
                if hists[year][ch][var].shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
                    continue
                # TODO: Add data
                if add_data:
                    fig, (ax, rax) = plt.subplots(nrows=2,
                                                  ncols=1,
                                                  figsize=(8,8),
                                                  tight_layout=True,
                                                  gridspec_kw={"height_ratios": (3, 1)},
                                                  sharex=True
                                              )
                    fig.subplots_adjust(hspace=.07)
                    data = hists[year][ch][var][{"samples": get_simplified_label(data_label)}]
                    hep.histplot(data,
                                 ax=ax,
                                 histtype="errorbar", 
                                 color="k",
                                 yerr=True,
                                 label=get_simplified_label(data)
                                 )
                else:
                    fig, ax = plt.subplots(1, 1)

                # plot the background stacked
                try:
                    hep.histplot([x for x in hists[year][ch][var].stack(0)[1:]],   # the [1:] is there to skip the signal sample which is usually given first in the samples list
                                 ax=ax,
                                 stack=True,
                                 sort='yield',
                                 histtype="fill",
                                 label=[x for x in hists[year][ch][var].axes[0]][1:],
                                 )
                except:
                    print('No background samples to plot besides the signal')
                # plot the signal separately on the same plot
                signal = hists[year][ch][var][{"samples": get_simplified_label(signal_by_ch[ch],pfnano)}]
                # if not logy then scale the signal by 10 (?)
                if not logy:
                    signal = signal*10
                hep.histplot(signal,
                             ax=ax,
                             stack=True,
                             label=get_simplified_label(signal_by_ch[ch],pfnano),
                             color='red'
                )
                # add ratio plot if we have data
                if add_data: 
                    rax.errorbar(
                        x=[data.axes.value(i)[0] for i in range(len(data.values()))],
                        y=data.values() / np.sum([b.values() for b in bkg], axis=0),
                        fmt="ko",
                    )
                    
                if logy:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1)
                ax.set_title(f'{ch} channel')
                ax.legend()

                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/hists_{year}/{var}_{ch}.pdf')
                plt.close()

def make_norm(idir, vars_to_plot, years, channels,logy=True):
    for year in years:
        if not os.path.exists(f'{odir}/hists_{year}'):
            os.makedirs(f'{odir}/hists_{year}')
        for ch in channels:
            for var in vars_to_plot:
                if hists[year][ch][var].shape[0] == 0:
                    continue

                fig, ax = plt.subplots(1, 1)
                hep.histplot([x for x in hists[year][ch][var].stack(0)[1:]],
                             ax=ax,
                             stack=False,
                             density=True,
                             )
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/hists_{year}/{var}_{ch}_density.pdf')
                plt.close()

def main(args):
    odir = args.odir
    if not os.path.exists(odir):
        os.makedirs(odir)

    years = args.years.split(',')
    channels = args.channels.split(',')

    # get samples to make histograms
    if args.pfnano:
        f = open('configs/samples_pfnano.json')
    else:
        f = open('configs/samples.json')
    json_samples = json.load(f)
    f.close()

    samples = []
    for key, value in json_samples.items():
        if value == 1:
            samples.append(key)

    # get variables to plot
    f = open(args.vars)
    variables = json.load(f)
    f.close()

    vars_to_plot = []
    for key, value in variables.items():
        if value == 1:
            vars_to_plot.append(key)

    # make the histograms and save in pkl files
    make_hist(args.idir, odir, vars_to_plot, samples, years, channels, args.pfnano)

    # plot all process in stack
    make_stack(odir, vars_to_plot, years, channels, args.pfnano)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_plots.py --year 2017 --vars configs/vars.json --channels ele,mu,had --idir ../results/ --odir hists --pfnano True
    parser = argparse.ArgumentParser()
    parser.add_argument('--years',      dest='years',       default='2017',                     help="year")
    parser.add_argument("--pfnano",     dest='pfnano',      default=False,                      help="Run with pfnano",                     action=BoolArg)
    parser.add_argument('--vars',       dest='vars',        default="configs/vars.json",        help='path to json with variables to be plotted')
    parser.add_argument('--channels',   dest='channels',    default='ele,mu,had',               help='channels for which to plot this variable')
    parser.add_argument('--odir',       dest='odir',        default='hists',                    help="tag for output directory")
    parser.add_argument('--idir',       dest='idir',        default='../results/',              help="input directory with results")

    args = parser.parse_args()

    main(args)
