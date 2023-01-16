#!/usr/bin/python

import json
import time

import argparse
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
import os, glob, sys
import pickle

from coffea.lumi_tools import LumiData, LumiMask, LumiList
"""
This script computes the total luminosity using a single lumi_set.pkl file 
and a lumi.csv produced using the GoldenJson.
"""

def main():
    # you can load the output!
    import pickle
    with open("lumi_set.pkl", 'rb') as f:
        lumi_set = pickle.load(f)

    # combine the sets from the different datasets
    for i, dataset in enumerate(lumi_set.keys()):
        print(f"Retrieving lumi_set of {dataset}")
        if i == 0:
            lumis = lumi_set[dataset]
        else:
            lumis = lumis | lumi_set[dataset]

    # convert the set to a numpy 2d-array
    lumis = np.array(list(lumis))
    

    # make LumiList object
    lumi_list = LumiList(runs=lumis[:, 0], lumis=lumis[:, 1])

    # this csv was made using brilcalc and the GoldenJson2017
    # refer to https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
    lumidata = LumiData("lumi2017.csv")
    print(f"---> Total Lumi = {lumidata.get_lumi(lumi_list)}")

if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python compute_lumi.py

    main()
