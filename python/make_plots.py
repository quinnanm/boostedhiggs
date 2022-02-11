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
    'dr_jet_candlep': hist2.axis.Regular(520, 0, 1.5, name='var', label=r'$\Delta R(l, Jet)$'),
    'met':  hist2.axis.Regular(50, 0, 400, name='var', label='MET [GeV]'),
    'mt_lep_met':  hist2.axis.Regular(20, 0, 300, name='var', label=r'$m_T(lep, p_T^{miss})$ [GeV]'),
    'mu_mvaId': hist2.axis.Regular(20, -1, 1, name='var', label='Muon MVAID'),
    'leadingfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet $p_T$ [GeV]'),
    'leadingfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet $m_{sd}$ [GeV]'),
    'secondfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'2nd Jet $p_T$ [GeV]'),
    'secondfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'2nd Jet $m_{sd}$ [GeV]'),
    'bjets_ophem_leadingfj': hist2.axis.Regular(20, 0, 1, name='var', label=r'btagFlavB (opphem)'),
}

def get_simplified_label(sample):   # get simplified "alias" names of the samples for plotting purposes
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

def make_hist(idir,odir,vars_to_plot,samples,years,channels): # makes histograms and saves in pkl file
    hists = {}  # define a placeholder for all histograms 

    for year in years:
        # Get luminosity of year
        f = open('../data/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}
        
        for ch in channels: # initialize the histograms for the different channels and different variables  
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
                    f = open('../data/xsecs.json')
                    xsec = json.load(f)
                    f.close()
                    xsec = eval(str((xsec[sample])))  # because some xsections are given as string formulas in the xsecs.json 
                    print('- xsection of sample is', xsec)

                    # Get sum_sumgenweight of sample                                                                                                                                                                                                   
                    sum_sumgenweight = get_sum_sumgenweight(idir, year, sample)
                    
                    # Get overall weighting of events
                    xsec_weight = (xsec * luminosity[year]) / (sum_sumgenweight)
                    
                    for ch in channels:
                        parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed

                        for parquet_file in parquet_files:
                            data = pq.read_table(parquet_file).to_pandas()
                            
                            for var in vars_to_plot:
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

    print(hists)
    # store the hists variable
    with open(f'{odir}/hists.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)

def make_stack(odir,vars_to_plot,years,channels):
    signal_by_ch = {'ele': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                    'mu': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8',
                    'had': 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8', # NOTE: need to change this file
    }

    # load the hists
    print(f'{odir}/hists.pkl')
    with open(f'{odir}/hists.pkl', 'wb') as f:
        hists = pkl.load(f)
            
    # make the histogram plots in this directory
    # TODO: we will want combined plots for all years later too
    for year in years:
        if not os.path.exists(f'{odir}/hists_{year}'):
            os.makedirs(f'{odir}/hists_{year}')

        for ch in channels:
            for var in vars_to_plot:
                if hists[year][ch][var].shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
                    continue
                fig, ax = plt.subplots(1, 1)
                # TODO: Add data
                # plot the background stacked
                hep.histplot([x for x in hists[year][ch][var].stack(0)[1:]],   # the [1:] is there to skip the signal sample which is usually given first in the samples list
                             ax=ax,
                             stack=True,
                             histtype="fill",
                             label=[x for x in hists[year][ch][var].axes[0]][1:],
                             )
                # plot the signal separately on the same plot
                hep.histplot(hists[year][ch][var][{"samples": get_simplified_label(signal_by_ch[ch])}],
                             ax=ax,
                             stack=True,
                             label=get_simplified_label(signal),
                             color='red'
                             )
                ax.set_yscale('log')
                ax.set_title(f'{ch} channel')
                ax.legend()

                hep.cms.lumitext("{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'hists/hists_{year}/{var}_{ch}.pdf')
                plt.close()    

                    
def main(args):
    odir = 'hists/'+args.odir
    if not os.path.exists(odir):
        os.makedirs(odir)

    vars_to_plot = args.var.split(',')
    years = args.year.split(',')
    samples = args.sample.split(',')
    channels = args.channel.split(',')

    # make the histograms and save in pkl files
    make_hist(args.idir,odir,vars_to_plot,samples,years,channels)

    # plot all process in stack
    make_stack(odir,vars_to_plot,years,channels)

if __name__ == "__main__":
    # e.g.
    # run locally as: 
    # python make_plots.py --year 2017 --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8,TTToHadronic_TuneCP5_13TeV-powheg-pythia8 --idir ../results/ --odir TAG --var met,ht --channel ele,mu,had
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',    dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--sample',  dest='sample',     default=None,         help='sample name', required=True)
    parser.add_argument('--channel', dest='channel',    default='ele,mu,had', help='channels for which to plot this variable', required=True)
    parser.add_argument('--odir',    dest='odir',       default='',           help="tag for output directory", type=str)
    parser.add_argument('--idir',    dest='idir',       default='../results/',           help="input directory with results", type=str)
    parser.add_argument('--var',     dest='var',        required=True, 
                        choices=["met","ht" # common for all channels
                                 "lepton_pt","lep_isolation","met","ht","mt_lep_met","dr_jet_candlep", # common for ele and mu channels
                                 "mu_mvaId", # mu channel only
                                 "leadingfj_pt","leadingfj_msoftdrop","secondfj_pt","secondfj_msoftdrop","bjets_ophem_leadingfj" # had channel only
                             ],
                        help='variable to plot')
    args = parser.parse_args()

    main(args)
