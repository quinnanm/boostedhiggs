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
    year = "2016"
    year_mod = "APV"
    dir_ = f"/eos/uscms/store/user/fmokhtar/boostedhiggs/lumi_{year+year_mod}/"

    import pickle

    with open(f"{dir_}/lumi_set.pkl", "rb") as f:
        lumi_set = pickle.load(f)

    # # combine the sets from the different datasets
    # for i, dataset in enumerate(lumi_set.keys()):
    #     print(f"Retrieving lumi_set of {dataset}")
    #     if i == 0:
    #         lumis = lumi_set[dataset]
    #     else:
    #         lumis = lumis | lumi_set[dataset]

    # # convert the set to a numpy 2d-array
    # lumis = np.array(list(lumis))

    # # make LumiList object
    # lumi_list = LumiList(runs=lumis[:, 0], lumis=lumis[:, 1])

    # # this csv was made using brilcalc and the GoldenJson2017
    # # refer to https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
    # lumidata = LumiData(f"lumi{year}.csv")
    # print(f"---> Total Lumi = {lumidata.get_lumi(lumi_list)}")

    print("------------------------------------")

    lumis = {}

    # combine the sets from the different datasets
    for i, dataset in enumerate(lumi_set.keys()):
        if "Muon" in dataset:
            ch = "mu"
        else:
            ch = "ele"

        print(f"Retrieving lumi_set of {dataset}")
        if ch not in lumis.keys():
            lumis[ch] = lumi_set[dataset]
        else:
            lumis[ch] = lumis[ch] | lumi_set[dataset]

    lumi_list = {}
    for ch in ["ele", "mu"]:
        # convert the set to a numpy 2d-array
        lumis[ch] = np.array(list(lumis[ch]))

        # make LumiList object
        lumi_list[ch] = LumiList(runs=lumis[ch][:, 0], lumis=lumis[ch][:, 1])

        # this csv was made using brilcalc and the GoldenJson2017
        # refer to https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
        lumidata = LumiData(f"lumi{year}.csv")
        print(f"---> Lumi for {ch} channel = {lumidata.get_lumi(lumi_list[ch])}")
        print("------------------------------------")


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python compute_lumi.py

    main()
