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
from make_stacked_hists import make_stacked_hists

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def get_simplified_label(sample):   # get simplified "alias" names of the samples for plotting purposes
    f = open('../data/simplified_labels.json')
    name = json.load(f)
    f.close()
    return str(name[sample])


def get_sum_sumgenweight(year, sample):

    pkl_files = glob.glob(f'../results/{sample}/outfiles/*.pkl')  # get the pkl metadata of the pkl files that were processed
    sum_sumgenweight = 0

    for file in pkl_files:
        # load and sum the sumgenweight of each
        with open(file, 'rb') as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]['sumgenweight']
    return sum_sumgenweight


def get_axis(var):  # define the axes for the different variables to be plotted
    if var == 'lepton_pt':
        return hist2.axis.Regular(50, 0, 400, name='var', label=var)
    elif var == 'lep_isolation':
        return hist2.axis.Regular(20, 0, 3.5, name='var', label=var)
    elif var == 'ht':
        return hist2.axis.Regular(20, 180, 1500, name='var', label=var)
    elif var == 'dr_jet_candlep':
        return hist2.axis.Regular(520, 0, 1.5, name='var', label=var)
    elif var == 'met':
        return hist2.axis.Regular(50, 0, 400, name='var', label=var)
    else:
        return hist2.axis.Regular(50, 0, 400, name='var', label=var)


def main(args):
    if not os.path.exists(f'hists/'):
        os.makedirs(f'hists/')

    # get variables to plot
    if args.var == None:  # plot all variables if none is specefied
        vars = ['lepton_pt', 'lep_isolation', 'ht', 'dr_jet_candlep', 'met']
    else:
        vars = args.var.split(',')

    years = args.year.split(',')
    samples = args.sample.split(',')

    signal = 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8'
    channels = ['ele', 'mu', 'had']

    hists = {}  # define a placeholder for all histograms

    for year in years:
        # Get luminosity of year
        f = open('../data/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        if not os.path.exists(f'hists/hists_{year}'):
            os.makedirs(f'hists/hists_{year}')

        hists[year] = {}

        for ch in channels:  # initialize the histograms
            hists[year][ch] = {}

            for var in vars:
                sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)

                hists[year][ch][var] = hist2.Hist(
                    sample_axis,
                    get_axis(var),
                )

        # loop over the processed files and fill the histograms
        for i, sample in enumerate(samples):

            print('Processing sample', sample)
            pkl_files = glob.glob(f'../results/{sample}/outfiles/*.pkl')  # get list of files that were processed
            if not pkl_files:  # skip samples which were not processed
                print('- No processed files found... skipping sample...')
                continue

            # Get xsection of sample
            f = open('../data/xsecs.json')
            xsec = json.load(f)
            f.close()
            xsec = eval(str((xsec[sample])))  # because some xsections are given as string formulas in the xsecs.json
            print('- xsection of sample is', xsec)

            # Get sum_sumgenweight of sample
            sum_sumgenweight = get_sum_sumgenweight(year, sample)

            # Get overall weighting of events
            xsec_weight = (xsec * luminosity[year]) / (sum_sumgenweight)

            for ch in channels:
                parquet_files = glob.glob(f'../results/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed

                for i, parquet_file in enumerate(parquet_files):
                    data = pq.read_table(parquet_file).to_pandas()

                    for var in vars:
                        if var not in data.keys():
                            continue

                        variable = data[var].to_numpy()
                        event_weight = data['weight'].to_numpy()

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
                                samples=get_simplified_label(sample),
                                var=variable,
                                weight=event_weight * xsec_weight,
                            )

    # store the hists variable
    with open(f'hists/hists_{year}.pkl', 'wb') as f:  # saves the hists object
        pkl.dump(hists, f)

    # make the histogram plots
    for year in years:
        for ch in channels:
            for var in vars:
                if hists[year][ch][var].shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
                    continue
                fig, ax = plt.subplots(1, 1)
                # plot the background stacked
                hep.histplot([x for x in hists[year][ch][var].stack(0)[1:]],   # the [1:] is there to skip the signal sample which is usually given first in the samples list
                             ax=ax,
                             stack=True,
                             histtype="fill",
                             label=[x for x in hists[year][ch][var].axes[0]][1:],
                             )
                # plot the signal seperately on the same plot
                hep.histplot(hists[year][ch][var][{"samples": get_simplified_label(signal)}],
                             ax=ax,
                             stack=True,
                             label=get_simplified_label(signal),
                             color='red'
                             )
                ax.set_yscale('log')
                ax.set_title(f'{ch} channel')
                ax.legend()

                hep.cms.lumitext("2017 (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'hists/hists_{year}/{var}_{ch}.pdf')
                plt.close()


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_plots.py --year 2017 --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8,TTToHadronic_TuneCP5_13TeV-powheg-pythia8
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--sample',     dest='sample',     default=None,         help='sample name', required=True)
    parser.add_argument("--combine",  dest='combine_processed_files',  action=BoolArg, default=True, help="combine the processed files to make histograms")
    parser.add_argument('--var',     dest='var',     default=None,         help='variable to plot')
    args = parser.parse_args()

    main(args)
