#!/usr/bin/python

from utils import add_samples, data_by_ch, data_by_ch_2018

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


parser = argparse.ArgumentParser()
parser.add_argument('--channels', dest='channels',  default='ele,mu,had',  help='channels for which to plot this variable')
parser.add_argument('--dir',      dest='dir',       default='May7_2017',   help="tag for data directory")
parser.add_argument('--odir',     dest='odir',      default='counts',       help="tag for output directory")

args = parser.parse_args()

# define color dict for plotting different samples
color_dict = {'signal': 'red',
              'QCD': 'blue',
              'DYJets': 'green',
              'TTbar': 'purple',
              'SingleTop': 'yellow',
              'WJetsLNu': 'brown',
              'WQQ': 'orange',
              'ZQQ': 'magenta',
              'others': 'black'
              }


def compute_counts(channels, samples, odir):
    """
    Given a list of samples and channels, computes the counts of events in the processed .pq files.
    Weights the event by the 'tot_weight' column and then sums up the counts.
    """

    num_dict = {}
    for ch in channels:
        num_dict[ch] = {}
        print(f'For {ch} channel')

        for sample in samples:
            # check if sample is data to skip
            is_data = False
            for key in data_label.values():
                if key in sample:
                    is_data = True
            if is_data:
                print('sample is_data so skipping')
                continue

            combine = False
            print(f'processing {sample} sample')
            for single_key, key in add_samples.items():
                if key in sample:
                    if single_key not in num_dict.keys():
                        num_dict[ch][single_key] = 0
                    combine = True
                    break
            if not combine:
                num_dict[ch][sample] = 0

            # get list of parquet files that have been processed
            parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')

            if len(parquet_files) == 0:
                continue

            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                    continue
                if len(data) == 0:
                    continue

                if combine:
                    num_dict[ch][single_key] = num_dict[ch][single_key] + data['tot_weight'].sum()
                else:
                    num_dict[ch][sample] = num_dict[ch][sample] + data['tot_weight'].sum()

            print(f'number of events for {sample} is {num_dict[ch][sample]}')

    with open(f'./{odir}/num_dict.pkl', 'wb') as f:  # saves counts
        pkl.dump(num_dict, f)


if __name__ == "__main__":
    """
    e.g. run locally as
    python counting_script.py --dir May7_2016 --ch ele --odir counts
    """

    channels = args.channels.split(',')

    year = args.dir[-4:]
    idir = '/eos/uscms/store/user/cmantill/boostedhiggs/' + args.dir

    if year == '2018':
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch

    samples = os.listdir(f'{idir}')

    # make directory to hold counts
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    compute_counts(channels, samples, args.odir)
