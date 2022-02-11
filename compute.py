#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import nanoevents
from coffea import processor
import time

import argparse
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
import pandas as pd
import os


def main(args):
    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    samples = []
    for key, value in json_samples.items():
        if value == 1:
            samples.append(key)

    # make directory for output
    if not os.path.exists('./outfiles'):
        os.makedirs('./outfiles')

    channels = ["ele", "mu", "had"]
    job_name = '/' + str(args.starti) + '-' + str(args.endi)

    # read samples to submit
    fileset = {}
    if args.pfnano:
        fname = f"data/pfnanoindex_{args.year}.json"
    else:
        fname = f"data/fileset_{args.year}_UL_NANO.json"
    with open(fname, 'r') as f:
        if args.pfnano:
            files = json.load(f)[args.year]
            for subdir in files.keys():
                for key, flist in files[subdir].items():
                    if key in samples:
                        fileset[key] = ["root://cmsxrootd.fnal.gov/" + f for f in flist[args.starti:args.endi]]
        else:
            files = json.load(f)
            for s in samples:
                fileset[s] = ["root://cmsxrootd.fnal.gov/" + f for f in files[s][args.starti:args.endi]]

    for sample in fileset:
        print(sample)


if __name__ == "__main__":
    # e.g.
    # run locally as: python compute.py --year 2017 --sample ../

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--samples',    dest='samples',     default="python/configs/samples.json",     help='path to json with samples to be plotted')
    args = parser.parse_args()

    main(args)
