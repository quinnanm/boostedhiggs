#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle as pkl

# counts events stored in root files

counts = {}

for ch in ['had', 'mu', 'ele']:
    counts[ch] = {}
    print(f'For {ch} channel')

    # repo = '/eos/uscms/store/user/mequinna/boostedhiggs/combinetest_23may22/merged/2017/'
    repo = '/uscms/home/fmokhtar/nobackup/boostedhiggs/python/merged/2017/'

    for sample in os.listdir(repo):

        fname = repo + f'{sample}_{ch}_merged.root'

        if os.path.isfile(f"{fname}"):
            # load in uproot
            events = uproot.open(f"{fname}")
            # sum tot_weight
            counts[ch][sample] = events['Events']['tot_weight'].array(library="np").sum()
            print(f'number of events for {sample} is {num_dict[ch][sample]}')
    print(f'-----------------------------------------')

with open(f'./counts_from_root.pkl', 'wb') as f:  # saves the hists objects
    pkl.dump(counts, f)
