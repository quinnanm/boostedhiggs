#!/usr/bin/python

import json
import uproot
import time

import argparse
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
import pandas as pd
import os, glob, sys
import pickle

def main():
    
    # load the pkl outfiles
    in_dir = "/eos/uscms/store/user/fmokhtar/boostedhiggs/lumi_2017/"
    out_dir = "/eos/uscms/store/user/fmokhtar/boostedhiggs/lumi/"

    datasets = ["SingleElectron_Run2017B", 
    "SingleElectron_Run2017D", 
    "SingleElectron_Run2017F", 
    "SingleMuon_Run2017C", 
    "SingleMuon_Run2017E",
    "SingleElectron_Run2017C", 
    "SingleElectron_Run2017E", 
    "SingleMuon_Run2017B", 
    "SingleMuon_Run2017D", 
    "SingleMuon_Run2017F"]
    
    out_all = {}
    for dataset in datasets:
        out_all[dataset]
        print(dataset)
        
        pkl_files = glob.glob(in_dir + dataset + "/outfiles/*")

        for i, pkl_file in enumerate(pkl_files):

            # you can load the output!
            with open(pkl_file, 'rb') as f:
                out = pickle.load(f)[dataset]['2017']['lumilist']

            if i==0:
                out_all[dataset] = out
            else:
                out_all[dataset] = out_all[dataset] | out

    # save the out_all
    with open(f"{out_dir}/lumi_set.pkl", 'wb') as handle:
        pkl.dump(out_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python lumi_combine.py

    main()
