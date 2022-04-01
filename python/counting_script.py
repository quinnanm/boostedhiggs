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
                    data = data[data['fj_pt'] != -1]

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        num_events[year][single_sample][ch]['preselection'] = num_events[year][single_sample][ch]['preselection'] + len(data)
                        num_events[year][single_sample][ch]['dr'] = num_events[year][single_sample][ch]['dr'] + len(data[data["leptonInJet"] == 1])
                        num_events[year][single_sample][ch]['btagdr'] = num_events[year][single_sample][ch]['btagdr'] + len(data[data["anti_bjettag"] == 1][data["leptonInJet"] == 1])
                    else:
                        num_events[year][sample][ch]['preselection'] = num_events[year][sample][ch]['preselection'] + len(data)
                        num_events[year][sample][ch]['dr'] = num_events[year][sample][ch]['dr'] + len(data[data["leptonInJet"] == 1])
                        num_events[year][sample][ch]['btagdr'] = num_events[year][sample][ch]['btagdr'] + len(data[data["anti_bjettag"] == 1][data["leptonInJet"] == 1])

    with open(f'{odir}/counts.pkl', 'wb') as f:  # dump the counts for further plotting
        pkl.dump(num_events, f)


# def plot_counts(idir, odir, samples, years, channels):
#
#     # load the counts dictionary
#     with open(f'{odir}/counts.pkl', 'rb') as f:
#         num_events = pkl.load(f)
#         f.close()
#
#     for year in years:
#         for sample in num_events[year]:
#


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

    print(f'Counting events of processed samples after each cut')
    count_events(args.idir, args.odir, samples, years, channels)


if __name__ == "__main__":
    # e.g. run locally as
    # python counting_script.py --year 2017 --odir counts --channels ele --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                        help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                  help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='counts',                     help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                 help="input directory with results")

    args = parser.parse_args()

    main(args)
