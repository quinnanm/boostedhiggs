#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle as pkl
import argparse

# counts events stored in root files

parser = argparse.ArgumentParser()
parser.add_argument('--ch',       dest='ch',        default='ele,mu,had',  help='channels for which to plot this variable')
parser.add_argument('--dir',      dest='dir',       default='rootfiles/',   help="tag for data directory")

args = parser.parse_args()

if __name__ == "__main__":
    """
    e.g. run locally as
    python counting_from_root_b4_hadd.py --dir rootfiles/ --ch ele
    """

    channels = args.ch.split(',')

    for ch in channels:

        print(f'For {ch} channel')
        counts = {}
        # repo = '/eos/uscms/store/user/mequinna/boostedhiggs/combinetest_23may22/merged/2017/'
        # repo = '/uscms/home/fmokhtar/nobackup/boostedhiggs/python/merged/2017/'

        for sample in os.listdir(args.dir + ch):
            counts[sample] = 0
            for root_file in os.listdir(args.dir + ch + '/' + sample):
                print('sample')
                # load in uproot
                events = uproot.open(f"{args.dir + ch + '/' + sample + '/' + root_file}")
                # sum tot_weight
                counts[sample] = counts[sample] + events['Events']['tot_weight'].array(library="np").sum()

        print(f'number of events for {sample} is {counts[sample]}')
        print(f'-----------------------------------------')
