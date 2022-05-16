#!/usr/bin/python

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
    indir = '/eos/uscms/store/user/fmokhtar/boostedhiggs/' + args.dir

    # make directory to hold rootfiles
    outdir = f'./counts_{args.ch}/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f'processing {args.ch} channel')
    num_dict = {}
    for subdir, dirs, files in os.walk(indir):
        print(f'processing {subdir} sample')
        num = 0
        for file in files:
            # load files
            if (f'{args.ch}.parquet' in file):
                f = subdir + '/' + file
                # f=f.replace("/eos/uscms/","root://cmseos.fnal.gov//")
                outf = f
                print('prepping input file', f, '...')
                outname = outf[outf.rfind(year + '/') + 5:]
                outname = outdir + outname.strip('.' + filetype) + '.root'
                outname = outname.replace('/outfiles/', '_')

                # load parquet into dataframe
                print('loading dataframe...')
                table = pq.read_table(f)
                data = table.to_pandas()
                print('# input events:', len(data))
                if len(data) == 0:
                    print('no skimmed events. skipping')
                    continue

                num = num + len(data)
        num_dict[subdir] = num

    with open(f'{outdir}/num_dict.pkl', 'wb') as f:  # saves counts
        pkl.dump(num_dict, f)
