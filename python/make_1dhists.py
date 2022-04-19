#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
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


def make_1dhists(idir, odir, samples, years, channels, var, bins, range):
    """
    Makes 1D histograms

    Args:
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

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
                hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True),     # to combine different pt bins of the same process
                hist2.axis.StrCategory([], name='cuts', growth=True)
            )

        num_events = {}
        for cut in ['preselection', 'btag', 'dr', 'btagdr']:
            num_events[cut] = 0

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                print("------------------------------------------------------------")
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample', sample)
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
                    data = data[data[var] != -1]

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        hists[year][ch].fill(
                            data[var],
                            single_sample,  # combining all events under one name
                            cuts='preselection'
                        )
                        hists[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1],
                            single_sample,  # combining all events under one name
                            cuts='btag'
                        )
                        hists[year][ch].fill(
                            data[var][data["leptonInJet"] == 1],
                            single_sample,  # combining all events under one name
                            cuts='dr'
                        )
                        hists[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            single_sample,  # combining all events under one name
                            cuts='btagdr'
                        )
                    else:
                        hists[year][ch].fill(
                            data[var],
                            sample,
                            cuts='preselection'
                        )
                        hists[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1],
                            sample,
                            cuts='btag'
                        )
                        hists[year][ch].fill(
                            data[var][data["leptonInJet"] == 1],
                            sample,
                            cuts='dr'
                        )
                        hists[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            sample,
                            cuts='btagdr'
                        )

                    num_events['preselection'] = num_events['preselection'] + len(data[var])
                    num_events['btag'] = num_events['btag'] + len(data[var][data["anti_bjettag"] == 1])
                    num_events['dr'] = num_events['dr'] + len(data[var][data["leptonInJet"] == 1])
                    num_events['btagdr'] = num_events['btagdr'] + len(data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1])

                for cut in ['preselection', 'btag', 'dr', 'btagdr']:
                    print(f"Num of events after {cut} cut is: {num_events[cut]}")
    print("------------------------------------------------------------")

    with open(f'{odir}/1d_hists_{var}.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_1dhists(odir, years, channels, var, cut='preselection'):
    """
    Plots 1D histograms that were made by "make_1dhists" function

    Args:
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
        cut: the cut to apply when plotting the histogram... choices are ['preselection', 'dr', 'btag', 'btagdr']
    """

    print(f'plotting for {cut} cut')
    # load the hists
    with open(f'{odir}/1d_hists_{var}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    for year in years:
        # make directory to store stuff per year
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}/')
        if not os.path.exists(f'{odir}/plots_{year}/{var}'):
            os.makedirs(f'{odir}/plots_{year}/{var}')
        # make plots per channel
        for ch in channels:
            for sample in hists[year][ch].axes[1]:
                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': cut}], ax=ax)
                ax.set_xlabel(f"{var}")
                ax.set_title(f'{ch} channel for \n {sample} \n with {cut} cut')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var}/{ch}_{sample}_{cut}.pdf')
                plt.close()

                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': cut}], ax=ax)
                ax.set_xlabel(f"{var}")
                ax.set_title(f'{ch} channel for \n {sample} \n with {cut} cut')
                ax.set_yscale('log')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var}/{ch}_{sample}_{cut}.pdf')
                plt.close()


def plot_1dhists_compare_cuts(odir, years, channels, var):
    """
    Plots 1D histograms that were made by "make_1dhists" function,
    with all cuts shown on the same plot for comparison

    Args:
        var: the name of the variable to plot a 1D-histogram of... see the full list of choices in plot_configs/vars.json
    """

    print(f'plotting all cuts on same plot for comparison')

    # load the hists
    with open(f'{odir}/1d_hists_{var}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    for year in years:
        # make directory to store stuff per year
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}/')
        if not os.path.exists(f'{odir}/plots_{year}/{var}'):
            os.makedirs(f'{odir}/plots_{year}/{var}')
        # make plots per channel
        for ch in channels:
            for sample in hists[year][ch].axes[1]:
                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'preselection'}],  ax=ax, label='preselection')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btag'}],          ax=ax, label='preselection + btag')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'dr'}],            ax=ax, label='preselection + leptonInJet')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btagdr'}],        ax=ax, label='preselection + btag + leptonInJet')
                ax.set_xlabel(f"{var}")
                ax.set_title(f'{ch} channel for \n {sample}')
                ax.legend()
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var}/{ch}_{sample}_all_cuts_comparison.pdf')
                plt.close()

                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'preselection'}],  ax=ax, label='preselection')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btag'}],          ax=ax, label='preselection + btag')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'dr'}],            ax=ax, label='preselection + leptonInJet')
                hep.histplot(hists[year][ch][{'samples': sample, 'cuts': 'btagdr'}],        ax=ax, label='preselection + btag + leptonInJet')
                ax.set_xlabel(f"{var}")
                ax.set_title(f'{ch} channel for \n {sample}')
                ax.legend()
                ax.set_yscale('log')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var}/{ch}_{sample}_all_cuts_comparison_log.pdf')
                plt.close()


def main(args):
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    years = args.years.split(',')
    channels = args.channels.split(',')

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

    range = [args.start, args.end]

    print(f'Plotting {args.var} histogram')

    if args.make_hists:
        make_1dhists(args.idir, args.odir, samples, years, channels, args.var, args.bins, range)

    if args.plot_hists:
        for cut in ['preselection', 'dr', 'btag', 'btagdr']:
            plot_1dhists(args.odir, years, channels, args.var, cut)

        plot_1dhists_compare_cuts(args.odir, years, channels, args.var)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt:    python make_1dhists.py --year 2017 --odir hists/1dhists --channels ele --var lep_pt    --make_hists --plot_hists --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_fj_dr: python make_1dhists.py --year 2017 --odir hists/1dhists --channels ele --var lep_fj_dr --make_hists --plot_hists --bins 100 --start 0 --end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # fj_pt:     python make_1dhists.py --year 2017 --odir hists/1dhists --channels had --var fj0_pt    --make_hists --plot_hists --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists/1dhists',                    help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument('--var',             dest='var',         default='lep_pt',                           help='variable to plot')
    parser.add_argument('--bins',            dest='bins',        default=50,                                 help="binning of the first variable passed",                type=int)
    parser.add_argument('--start',           dest='start',       default=0,                                  help="starting range of the first variable passed",         type=int)
    parser.add_argument('--end',             dest='end',         default=1,                                  help="end range of the first variable passed",              type=int)
    parser.add_argument("--make_hists",      dest='make_hists',     action='store_true',                          help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',     action='store_true',                          help="Plot the hists")

    args = parser.parse_args()

    main(args)
