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

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def main(args):
    years = args.year.split(',')

    # preprocessing step before making histograms
    # combines the parquet files while keeping track of the metadata and saves it as pkl files
    if args.combine_processed_files:

        samples = args.sample.split(',')

        channels = ['ele', 'mu', 'had']
        years = ['2017']
        luminosity = {}

        for year in years:

            if not os.path.exists(f'hists/'):
                os.makedirs(f'hists/')

            if not os.path.exists(f'hists/hists_{year}'):
                os.makedirs(f'hists/hists_{year}')

            # Get luminosity of year
            f = open('../data/luminosity.json')
            luminosity = json.load(f)
            f.close()
            print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

            xsec = {}
            xsec_weight = {}

            data_all = {}

            for sample in samples:
                print('Processing sample', sample)
                pkl_files = glob.glob(f'../results/{sample}/outfiles/*.pkl')  # get list of metadata pkl files that need to be processed
                if not pkl_files:  # skip samples which were not processed
                    print('- No processed files found... skipping sample...')
                    continue

                # Get xsection of sample
                f = open('../data/xsecs.json')
                data = json.load(f)
                f.close()
                xsec[sample] = eval(str((data[sample])))  # because some xsections are given as string formulas in the xsecs.json
                print('- xsection of sample is', xsec[sample])

                # define some initializers
                sum_sumgenweight = {}
                sum_sumgenweight[sample] = 0
                data_all[sample] = {}

                for ch in channels:
                    parquet_files = glob.glob(f'../results/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that need to be processed
                    for i, parquet_file in enumerate(parquet_files):
                        tmp = pq.read_table(parquet_file).to_pandas()
                        if i == 0:
                            data = tmp
                        else:
                            data = pd.concat([data, tmp], ignore_index=True)

                        # load and sum the sumgenweight of each
                        with open(pkl_files[i], 'rb') as f:
                            metadata = pkl.load(f)
                        sum_sumgenweight[sample] = sum_sumgenweight[sample] + metadata[sample][year]['sumgenweight']
                    data_all[sample][ch] = data

                print('- # of files processed is', i + 1)

                xsec_weight[sample] = (xsec[sample] * luminosity[year]) / (sum_sumgenweight[sample])

            with open(f'hists/hists_{year}/samples.pkl', 'wb') as f:  # save a list of the samples covered
                pkl.dump(samples, f)
            with open(f'hists/hists_{year}/data_all.pkl', 'wb') as f:  # save a variable that contains all the events post selection
                pkl.dump(data_all, f)
            with open(f'hists/hists_{year}/xsec_weight.pkl', 'wb') as f:  # saves a variable that contains the xsec weight of the sample
                pkl.dump(xsec_weight, f)

    else:
        for year in years:
            with open(f'hists/hists_{year}/samples.pkl', 'rb') as f:
                samples = pkl.load(f)
            with open(f'hists/hists_{year}/data_all.pkl', 'rb') as f:
                data_all = pkl.load(f)
            with open(f'hists/hists_{year}/xsec_weight.pkl', 'rb') as f:
                xsec_weight = pkl.load(f)

            if args.var == None:  # plot all variables if none is specefied
                vars = ['lepton_pt', 'lep_isolation', 'ht', 'dr_jet_candlep', 'met']
            else:
                vars = args.var.split(',')
            for var in vars:
                if var == 'lepton_pt':
                    make_stacked_hists('lepton_pt', 50, 0, 400, ['ele', 'mu'], samples, data_all, xsec_weight, year, r'Lepton $p_T$ [GeV]')
                if var == 'lep_isolation':
                    make_stacked_hists('lep_isolation', 20, 0, 3.5, ['ele', 'mu'], samples, data_all, xsec_weight, year, r'Lepton isolation')
                if var == 'ht':
                    make_stacked_hists('ht', 20, 180, 1500, ['had'], samples, data_all, xsec_weight, year, r'HT [GeV]')
                if var == 'dr_jet_candlep':
                    make_stacked_hists('dr_jet_candlep', 20, 0, 1.5, ['ele', 'mu'], samples, data_all, xsec_weight, year, r'dr_jet_candlep')
                if var == 'met':
                    make_stacked_hists('met', 20, 0, 1.5, ['ele', 'mu', 'had'], samples, data_all, xsec_weight, year, r'dr_jet_candlep')


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
