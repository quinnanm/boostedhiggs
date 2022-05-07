#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle as pkl

# load a root file into coffea-friendly NanoAOD structure

samples = os.listdir("./rootfiles/")
num_dict = {}

for ch in ['had', 'mu', 'ele']:
    num_dict[ch] = {}
    print(f'For {ch} channel')
    for sample in samples:
        if os.path.isfile(f"./rootfiles/{sample}/{sample}_{ch}.root"):
            f = uproot.open(f"./rootfiles/{sample}/{sample}_{ch}.root")
            num = f['Events'].num_entries  # checks number of events per file
            print(f'number of events for {sample} is {num}')
            num_dict[ch][sample] = num
    print(f'-----------------------------------------')

with open(f'./rootfiles/num_dict.pkl', 'wb') as f:  # saves the hists objects
    pkl.dump(num_dict, f)
