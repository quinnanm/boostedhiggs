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
parser.add_argument('--ch',             dest='ch',              default='ele',                          help='channel for which to plot this variable')
parser.add_argument('--dir',            dest='dir',               default='Apr20_2016',                              help="tag for output directory")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. run locally as
    python counting_script.py --dir Apr20_2016 --ch ele
    """

    year = args.dir[-4:]
    idir = '/eos/uscms/store/user/fmokhtar/boostedhiggs/' + args.dir

    if year == '2018':
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch

    # make directory to hold rootfiles
    outdir = f'./counts_{args.ch}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    samples = os.listdir(f'{idir}')
    num_dict = {}

    print(f'processing {args.ch} channel')
    for sample in samples:
        # check if sample is data to skip
        is_data = False
        for key in data_label.values():
            if key in sample:
                is_data = True
        if is_data:
            continue

        combine = False
        print(f'processing {sample} sample')
        for single_key, key in add_samples.items():
            if key in sample:
                if single_key not in num_dict.keys():
                    num_dict[single_key] = 0
                combine = True
                break
        if not combine:
            num_dict[sample] = 0

        # get list of parquet files that have been processed
        parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{args.ch}.parquet')

        if len(parquet_files) == 0:
            continue

        print('combine', combine)

        for i, parquet_file in enumerate(parquet_files):
            try:
                data = pq.read_table(parquet_file).to_pandas()
            except:
                print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                continue
            if len(data) == 0:
                continue
            print(data['tot_weight'])
            print(data['tot_weight'].sum())
            if combine:
                num_dict[single_key] = num_dict[single_key] + data['tot_weight'].sum()
            else:
                num_dict[sample] = num_dict[sample] + data['tot_weight'].sum()

    with open(f'{outdir}/num_dict.pkl', 'wb') as f:  # saves counts
        pkl.dump(num_dict, f)
