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


def count_events(idir, odir, samples, years, channels):
    """
    Counts events oprocessed per sample after each cut

    Args:
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    print(f'Counting events of processed samples after each cut')

    num_events = {}
    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        # initialize a num_events dictionary to count events per sample after each cut (but have to merge same process different bins samples)
        num_events[year] = {}
        for ch in channels:
            for sample in samples[year][ch]:
                single_sample = None
                for single_key, key in add_samples.items():
                    if key in sample:
                        single_sample = single_key

                if single_sample is not None:
                    num_events[year][single_sample] = {}
                    num_events[year][single_sample][ch] = {}
                    for cut in ['preselection', 'dr', 'btagdr']:
                        num_events[year][single_sample][ch][cut] = 0
                else:
                    num_events[year][sample] = {}
                    num_events[year][sample][ch] = {}
                    for cut in ['preselection', 'dr', 'btagdr']:
                        num_events[year][sample][ch][cut] = 0

        for ch in channels:
            for sample in samples[year][ch]:
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample {sample} with {len(parquet_files)} number of files')

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue
                    if len(data) == 0:
                        continue

                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    try:
                        data = data[data['fj_pt'] != -1]
                    except:
                        data = data[data['fj0_pt'] != -1]

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        num_events[year][single_sample][ch]['preselection'] = num_events[year][single_sample][ch]['preselection'] + len(data)
                        if ch != 'had':
                            num_events[year][single_sample][ch]['dr'] = num_events[year][single_sample][ch]['dr'] + len(data[data["leptonInJet"] == 1])
                            num_events[year][single_sample][ch]['btagdr'] = num_events[year][single_sample][ch]['btagdr'] + len(data[data["anti_bjettag"] == 1][data["leptonInJet"] == 1])
                    else:
                        num_events[year][sample][ch]['preselection'] = num_events[year][sample][ch]['preselection'] + len(data)
                        if ch != 'had':
                            num_events[year][sample][ch]['dr'] = num_events[year][sample][ch]['dr'] + len(data[data["leptonInJet"] == 1])
                            num_events[year][sample][ch]['btagdr'] = num_events[year][sample][ch]['btagdr'] + len(data[data["anti_bjettag"] == 1][data["leptonInJet"] == 1])
                if ch == 'had':
                    print(f"Sample {sample} has {num_events[year][single_sample][ch]['preselection']} events after cuts")

    with open(f'{odir}/counts_{channels[0]}.pkl', 'wb') as f:  # dump the counts for further plotting
        pkl.dump(num_events, f)


def plot_counts(odir, years, channels):
    """
    Plot the counts that were computed using the "count_events" function
    """

    print(f'Plotting the counts of each sample after each cut')

    # load the counts dictionary
    with open(f'{odir}/counts_{channels[0]}.pkl', 'rb') as f:
        num_events = pkl.load(f)
        f.close()

    for year in years:
        # make directories to hold plots
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}/')

        for sample in num_events[year].keys():
            for ch in channels:
                counts = []
                for key, value in (num_events[year][sample][ch]).items():
                    counts.append(value)

                fig, ax = plt.subplots(figsize=(8, 5))
                plt.bar([0, 2, 4],
                        counts,
                        tick_label=['preselection', 'preselection + \n dr', 'preselection + \n dr + \n btag'],
                        log=True)
                ax.set_xlabel(f"cut")
                ax.set_title(f'{ch} channel for \n {sample}')
                plt.savefig(f'{odir}/plots_{year}/counts_{ch}_{sample}.pdf')


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

    if args.compute_counts:
        count_events(args.idir, args.odir, samples, years, channels)

    if args.plot_counts:
        plot_counts(args.odir, years, channels)


if __name__ == "__main__":
    # e.g. run locally as
    # python counting_script.py --year 2017 --odir counts --channels had --compute_counts --plot_counts --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',              default='2017',                                help="year")
    parser.add_argument('--samples',         dest='samples',            default="plot_configs/samples_pfnano.json",    help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',           default='ele,mu,had',                          help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',               default='counts',                              help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',               default='../results/',                         help="input directory with results")
    parser.add_argument("--compute_counts",  dest='compute_counts',     action='store_true',                           help="Compute the counts")
    parser.add_argument("--plot_counts",     dest='plot_counts',        action='store_true',                           help="Plot the counts")

    args = parser.parse_args()

    main(args)
