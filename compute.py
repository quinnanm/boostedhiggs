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

    # read samples to submit
    fileset = {}
    fname = f"data/fileset_{args.year}_UL_NANO.json"
    with open(fname, 'r') as f:
        files = json.load(f)
        for s in samples:
            fileset[s] = ["root://cmsxrootd.fnal.gov/" + f for f in files[s]]
    import uproot
    for sample in fileset:
        for file in fileset[sample]:
            print(file)
            # d = uproot.open(file)

        # print()


if __name__ == "__main__":
    # e.g.
    # run locally as: python compute.py --year 2017 --sample ../

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--samples',    dest='samples',     default="python/configs/samples.json",     help='path to json with samples to be plotted')
    args = parser.parse_args()

    main(args)
