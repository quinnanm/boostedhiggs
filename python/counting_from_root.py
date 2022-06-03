#import fastparquet
import os
import uproot
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle as pkl

# counts events stored in root files

num_dict = {}

for ch in ['had', 'mu', 'ele']:
    num_dict[ch] = {}
    print(f'For {ch} channel')
    for sample in ['QCD', 'data', 'signal', 'signal2', 'ttbar', 'wjets', 'other']:
        if os.path.isfile(f"/eos/uscms/store/user/mequinna/boostedhiggs/combinetest_23may22/merged/2017/{sample}_ele_merged.root"):
            # load in uproot
            f = uproot.open(f"/eos/uscms/store/user/mequinna/boostedhiggs/combinetest_23may22/merged/2017/{sample}_ele_merged.root")
            # sum tot_weight
            num = events['Events']['tot_weight'].array(library="np").sum()
            print(f'number of events for {sample} is {num}')
            num_dict[ch][sample] = num
    print(f'-----------------------------------------')

with open(f'./counts_from_root.pkl', 'wb') as f:  # saves the hists objects
    pkl.dump(num_dict, f)
