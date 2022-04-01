#!/usr/bin/python

from axes import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
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


def get_sum_sumgenweight(idir, year, sample):
    pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get the pkl metadata of the pkl files that were processed
    sum_sumgenweight = 1  # TODO why not 0
    for file in pkl_files:
        # load and sum the sumgenweight of each
        with open(file, 'rb') as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]['sumgenweight']
    return sum_sumgenweight


def make_1dhists_ratio(idir, odir, samples, years, channels, vars, bins, start, end):

    # for readability
    x = vars[0]
    y = vars[1]

    hists = {}
    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}

        for ch in channels:  # initialize the histograms for the different channels and different variables
            hists[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, start, end, name=x + '/' + y, label=x + '/' + y, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True),
                hist2.axis.StrCategory([], name='cuts', growth=True)
            )

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                num_events = 0
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of {sample}')
                else:
                    print(f'No processed files for {sample} are found')

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue
                    if len(data) == 0:
                        continue

                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    data = data[data[x] != -1]
                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    data = data[data[y] != -1]

                    try:
                        event_weight = data['weight'].to_numpy()
                        # Find xsection if MC
                        f = open('../fileset/xsec_pfnano.json')
                        xsec = json.load(f)
                        f.close()
                        xsec = eval(str((xsec[sample])))

                        # Get overall weighting of events
                        xsec_weight = (xsec * luminosity[year]) / (get_sum_sumgenweight(idir, year, sample))

                    except:  # for data
                        data['weight'] = 1  # for data fill a weight column with ones
                        xsec_weight = 1

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        hists[year][ch].fill(
                            data[x] / data[y],
                            single_sample,
                            cuts='preselection',
                            weight=xsec_weight * data['weight']  # combining all events under one name
                        )
                        hists[year][ch].fill(
                            data[x][data["anti_bjettag"] == 1] / data[y][data["anti_bjettag"] == 1],
                            single_sample,
                            cuts='btag',
                            weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1]  # combining all events under one name
                        )
                        hists[year][ch].fill(
                            data[x][data["leptonInJet"] == 1] / data[y][data["leptonInJet"] == 1],
                            single_sample,
                            cuts='dr',
                            weight=xsec_weight * data['weight'][data["leptonInJet"] == 1]  # combining all events under one name
                        )
                        hists[year][ch].fill(
                            data[x][data["anti_bjettag"] == 1][data["leptonInJet"] == 1] / data[y][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            single_sample,
                            cuts='btagdr',
                            weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1][data["leptonInJet"] == 1]  # combining all events under one name
                        )
                    else:
                        hists[year][ch].fill(
                            data[x] / data[y],
                            sample,
                            cuts='preselection',
                            weight=xsec_weight * data['weight']
                        )
                        hists[year][ch].fill(
                            data[x][data["anti_bjettag"] == 1] / data[y][data["anti_bjettag"] == 1],
                            sample,
                            cuts='btag',
                            weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1]
                        )
                        hists[year][ch].fill(
                            data[x][data["leptonInJet"] == 1] / data[y][data["leptonInJet"] == 1],
                            sample,
                            cuts='dr',
                            weight=xsec_weight * data['weight'][data["leptonInJet"] == 1]
                        )
                        hists[year][ch].fill(
                            data[x][data["anti_bjettag"] == 1][data["leptonInJet"] == 1] / data[y][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            sample,
                            cuts='btagdr',
                            weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1][data["leptonInJet"] == 1]
                        )

                    num_events = num_events + len(data[x])
                print(f"Num of events is {num_events}")

    with open(f'{odir}/1d_hists_ratio.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_1dhists_ratio(odir, years, channels, vars, cut):

    print(f'plotting for {cut} cut')
    # load the hists
    with open(f'{odir}/1d_hists_ratio.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    for year in years:
        # make directories to hold plots
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}')
        if not os.path.exists(f'{odir}/plots_{year}/ratio_{vars[0]}_vs_{vars[1]}'):
            os.makedirs(f'{odir}/plots_{year}/ratio_{vars[0]}_vs_{vars[1]}')
        # make plots per channel
        for ch in channels:
            for sample in hists[year][ch].axes[1]:
                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': cut}], ax=ax)
                ax.set_xlabel(f"{vars[0]}/{vars[1]}")
                ax.set_title(f'{ch} channel for \n {sample} \n with {cut} cut')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/ratio_{vars[0]}_{vars[1]}/{ch}_{sample}_{cut}.pdf')
                plt.close()


def plot_1dhists_ratio_compare_cuts(odir, years, channels, vars):

    print(f'plotting all cuts on same plot for comparison')

    # load the hists
    with open(f'{odir}/1d_hists_ratio.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    for year in years:
        # make directories to hold plots
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}')
        if not os.path.exists(f'{odir}/plots_{year}/ratio_{vars[0]}_vs_{vars[1]}'):
            os.makedirs(f'{odir}/plots_{year}/ratio_{vars[0]}_vs_{vars[1]}')
        # make plots per channel
        for ch in channels:
            for sample in hists[year][ch].axes[1]:
                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'preselection'}],  ax=ax, label='preselection')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btag'}],          ax=ax, label='preselection + btag')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'dr'}],            ax=ax, label='preselection + leptonInJet')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btagdr'}],        ax=ax, label='preselection + btag + leptonInJet')
                ax.set_xlabel(f"{vars[0]}/{vars[1]}")
                ax.set_title(f'{ch} channel for \n {sample}')
                ax.legend()
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/ratio_{vars[0]}_{vars[1]}/{ch}_{sample}_all_cuts_comparison.pdf')
                plt.close()


def main(args):
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    years = args.years.split(',')
    channels = args.channels.split(',')
    vars = args.vars.split(',')

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    print(f'Making histograms of {vars[0]}/{vars[1]}')

    if args.make_hists:
        make_1dhists_ratio(args.idir, args.odir, samples, years, channels, vars, args.bins, args.start, args.end)

    if args.plot_hists:
        for cut in ['preselection', 'dr', 'btag', 'btagdr']:
            plot_1dhists_ratio(args.odir, years, channels, vars, cut)

        plot_1dhists_compare_cuts(args.odir, years, channels, vars)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt vs lep_iso:   python make_1dhists_ratio.py --year 2017 --odir hists/2dplots --channels ele --vars lep_pt,lep_isolation --make_hists --plot_hists --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs dR:        python make_1dhists_ratio.py --year 2017 --odir hists/2dplots --channels ele --vars lep_pt,lep_fj_dr     --make_hists --plot_hists --bins 100 --start 0 --end 500 --cut dr --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs mt:        python make_1dhists_ratio.py --year 2017 --odir hists/2dplots --channels ele --vars lep_pt,lep_met_mt    --make_hists --plot_hists --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs fj_pt:     python make_1dhists_ratio.py --year 2017 --odir hists/2dplots --channels ele --vars lep_pt,fj_pt         --make_hists --plot_hists --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                                 help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json",     help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                           help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists/2dplots',                        help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                          help="input directory with results")
    parser.add_argument('--vars',            dest='vars',        default='lep_pt,lep_isolation',                 help='channels for which to plot this variable')
    parser.add_argument('--bins',          dest='bins',      default=50,                                     help="binning of the first variable passed",                type=int)
    parser.add_argument('--start',         dest='start',     default=0,                                      help="starting range of the first variable passed",         type=int)
    parser.add_argument('--end',           dest='end',       default=1,                                      help="end range of the first variable passed",              type=int)
    parser.add_argument("--make_hists",      dest='make_hists',     action='store_true',                          help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',     action='store_true',                          help="Plot the hists")

    args = parser.parse_args()

    main(args)
