# script for converting pandas dataframe to rootfile
#import fastparquet
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
import uproot
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.options.mode.chained_assignment = None  # default='warn'


parser = argparse.ArgumentParser()
parser.add_argument('--ch',       dest='ch',        default='ele,mu,had',  help='channels for which to plot this variable')
parser.add_argument('--dir',      dest='dir',       default='May7_2017',   help="tag for data directory")
parser.add_argument('--odir',     dest='odir',      default='rootfiles',   help="tag for output directory")

args = parser.parse_args()


def compute_counts(channels, samples, idir, outdir, data_label):
    """
    Given a list of samples and channels, reads the .pq files and saves them as .root files.
    """

    for ch in channels:
        if not os.path.exists(outdir + ch):
            os.makedirs(outdir + ch)

        print(f'For {ch} channel')

        for sample in samples:
            print(f'For {sample} sample')

            # check if sample is data to skip
            is_data = False
            for key in data_label.values():
                if key in sample:
                    is_data = True
            if is_data:
                continue

            # combine samples under one key (e.g. all DY pt bins should be saved under the same DY directory)
            combine = False
            for single_key, key in add_samples.items():
                if key in sample:
                    combine = True
                    break

            if combine:
                dir_name = single_key
            else:
                dir_name = sample

            if not os.path.exists(outdir + ch + '/' + dir_name):
                os.makedirs(outdir + ch + '/' + dir_name)

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

                # load parquet into dataframe
                print('loading dataframe...')
                table = pq.read_table(parquet_file)
                data = table.to_pandas()
                print('# input events:', len(data))

                if len(data) == 0:
                    print('no skimmed events. skipping')
                    continue

                # drop AK columns
                for key in data.keys():
                    if data[key].dtype == 'object':
                        data.drop(columns=[key], inplace=True)

                _, tail = os.path.split(parquet_file)    # get the file name from full path

                outname = outdir + ch + '/' + dir_name + '/' + tail[:-8] + '_{sample}' + '.root'  # the [:-8] slice removes the .parquet extension (to replace it with a .root extension)
                with uproot.recreate(outname) as file:
                    file['Events'] = pd.DataFrame(data)

                print('Wrote rootfile ', outname)


if __name__ == "__main__":
    """
    e.g. run locally as
    python dataframe2root.py --dir May7_2017 --ch ele,mu,had --odir rootfiles
    """

    channels = args.ch.split(',')

    year = args.dir[-4:]    # get the year from the --dir argument being passed
    idir = '/eos/uscms/store/user/cmantill/boostedhiggs/' + args.dir

    if year == '2018':
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch

    samples = os.listdir(f'{idir}')

    # make directory to hold counts
    outdir = args.odir + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    compute_counts(channels, samples, idir, outdir, data_label)
