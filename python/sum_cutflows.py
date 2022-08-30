#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
from utils import get_simplified_label, get_sum_sumgenweight
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import sys
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings

warnings.filterwarnings("ignore", message="Found duplicate branch ")


def sum_cutflows(year, channels, idir, odir, samples):
    """
    Counts signal and background at different working points of a cut

    Args:
        year: string that represents the year the processed samples are from
        channels: list of channels... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    for ch in channels:
        if ch == "had":
            cut_keys = [
                "none",
                "trigger",
                "metfilters",
                "oneFatjet",
                "fatjetKin",
                "fatjetSoftdrop",
                "qcdrho",
                "met",
                "antibjettag",
            ]
        else:
            cut_keys = [
                "none",
                "trigger",
                "metfilters",
                "fatjetKin",
                "ht",
                "antibjettag",
                "leptonInJet",
                "leptonKin",
                "oneLepton",
                "notaus",
            ]
        cut_values = {}

        # loop over the samples
        for sample in samples[year][ch]:
            print(f"Processing sample {sample}")
            # skip data samples
            is_data = False
            for key in data_by_ch.values():
                if key in sample:
                    is_data = True
            if is_data:
                continue

            # check if the sample was processed
            pkl_dir = f"{idir}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)  #
            if not pkl_files:  # skip samples which were not processed
                continue

            for i, pkl_file in enumerate(pkl_files):

                single_sample = None
                for single_key, key in add_samples.items():
                    if key in sample:
                        single_sample = single_key
                if year == "Run2" and is_data:
                    single_sample = "Data"
                if single_sample is not None:
                    sample_to_use = single_sample
                else:
                    sample_to_use = sample

                if sample_to_use not in cut_values.keys():
                    cut_values[sample_to_use] = [0] * len(cut_keys)  # initialize

                with open(pkl_file, "rb") as f:
                    metadata = pkl.load(f)

                print("0", metadata)
                print("1", metadata[sample])
                print("2", metadata[sample][year])
                print("3", metadata[sample][year]["cutflows"])

                cutflows = metadata[sample][year]["cutflows"][ch]
                print("cutflows", cutflows)
                cutflows_sorted = sorted(cutflows.items(), key=lambda x: x[1], reverse=True)

                for i, elem in enumerate(cutflows_sorted):
                    cut_values[sample_to_use][i] += elem[1]

            print("------------------------------------------------------------")

        with open(f"{odir}/cut_values_{ch}.pkl", "wb") as f:  # saves the objects
            pkl.dump(cut_values, f)


def main(args):

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    channels = args.channels.split(",")

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    samples[args.year] = {}
    for ch in channels:
        samples[args.year][ch] = []
        for key, value in json_samples[args.year][ch].items():
            if value == 1:
                samples[args.year][ch].append(key)

    sum_cutflows(args.year, channels, args.idir, args.odir, samples)


if __name__ == "__main__":
    # e.g. run locally as
    # python sum_cutflows.py --year 2017 --odir cutflows --channels ele,mu,had --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/Aug11_2017

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year")
    parser.add_argument(
        "--samples",
        dest="samples",
        default="plot_configs/samples_pfnano_value.json",
        help="path to json with samples to be plotted",
    )
    parser.add_argument("--channels", dest="channels", default="ele,mu,had", help="channels for which to plot this variable")
    parser.add_argument(
        "--odir", dest="odir", default="hists", help="tag for output directory... will append '_{year}' to it"
    )
    parser.add_argument("--idir", dest="idir", default="../results/", help="input directory with results")

    args = parser.parse_args()

    main(args)
