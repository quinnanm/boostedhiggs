#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle as pkl

# counts events stored in root files


for ch in ['had', 'mu', 'ele']:

    print(f'For {ch} channel')

    # repo = '/eos/uscms/store/user/mequinna/boostedhiggs/combinetest_23may22/merged/2017/'
    repo = '/uscms/home/fmokhtar/nobackup/boostedhiggs/python/merged/2017'

    for merged_file in os.listdir(repo):

        # load in uproot
        events = uproot.open(f"{merged_file}")
        # sum tot_weight
        counts = events['Events']['tot_weight'].array(library="np").sum()
        print(f'number of events for {merged_file[:-12]} is {counts}')
    print(f'-----------------------------------------')
