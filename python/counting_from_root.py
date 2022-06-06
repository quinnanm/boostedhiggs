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
parser.add_argument('--dir',      dest='dir',       default='/uscms/home/fmokhtar/nobackup/boostedhiggs/python/merged/2017/',   help="tag for data directory")

args = parser.parse_args()

if __name__ == "__main__":
    """
    e.g. run locally as
    python counting_from_root.py --dir merged/2017/ --ch ele
    """

    channels = args.ch.split(',')

    for ch in channels:

        print(f'For {ch} channel')

        for merged_file in os.listdir(args.dir):
            if ch not in merged_file:
                continue
            # load in uproot
            events = uproot.open(f"{args.dir+merged_file}")
            # sum tot_weight
            counts = events['Events']['tot_weight'].array(library="np").sum()
            print(f'number of events for {merged_file[:-12]} is {counts}')  # the slice [:-12] just removes the parquet extension to display the sample name without it
        print(f'-----------------------------------------')
