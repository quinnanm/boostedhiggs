#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import nanoevents
from coffea import processor
import time
import glob

import argparse
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
import pandas as pd
import os


def get_simplified_label(sample):   # get simplified "alias" names of the samples for plotting purposes
    f = open('data/simplified_labels.json')
    name = json.load(f)
    f.close()
    return str(name[sample])


def main(args):

    years = args.years.split(',')
    idir = args.idir

    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    samples = []
    for key, value in json_samples.items():
        if value == 1:
            samples.append(key)

    for year in years:
        d = 0
        for sample in samples:
            c = 0
            pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get list of files that were processed
            if not pkl_files:  # skip samples which were not processed
                print('- No processed files found... skipping sample...')
                continue
            for file in pkl_files:
                # store the hists variable
                with open(f'{file}', 'rb') as f:
                    variable = pkl.load(f)
                    f.close()
                c = c + variable[sample][year]['cutflows']['ele']['all']
            print(f'{get_simplified_label(sample)} has {c} events')
            d = d + c
        print(f'Total # of events is {d}')


if __name__ == "__main__":
    # e.g.
    # run locally as: python compute_amount_of_data.py --year 2017

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',       dest='years',       default='2017',       help="year", type=str)
    parser.add_argument('--samples',    dest='samples',     default="python/configs/samples.json",     help='path to json with samples to be plotted')
    parser.add_argument('--idir',       dest='idir',        default='results/',              help="input directory with results", type=str)
    args = parser.parse_args()

    main(args)
