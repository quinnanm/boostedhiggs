#!/usr/bin/python

from utils import add_samples, data_by_ch, data_by_ch_2018

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ch",
    dest="ch",
    default="ele,mu,had",
    help="channels for which to plot this variable",
)
parser.add_argument(
    "--dir", dest="dir", default="May7_2017", help="tag for data directory"
)
parser.add_argument(
    "--odir", dest="odir", default="counts", help="tag for output directory"
)

args = parser.parse_args()


def compute_counts(channels, samples, odir, data_label):
    """
    Given a list of samples and channels, computes the counts of events in the processed .pq files.
    Weights the event by the 'tot_weight' column and then sums up the counts.
    """

    num_dict = {}
    for ch in channels:
        num_dict[ch] = {}
        print(f"For {ch} channel")

        for sample in samples:
            # check if sample is data to skip
            is_data = False
            for key in data_label.values():
                if key in sample:
                    is_data = True
            if is_data:
                continue

            # combine samples under one key (e.g. all DY pt bins should be combined under one single_key)
            combine = False
            for single_key, key in add_samples.items():
                if key in sample:
                    combine = True
                    break

            if (
                combine and single_key not in num_dict[ch].keys()
            ):  # if the counts for the combined samples has not been intialized yet
                num_dict[ch][single_key] = 0
            elif not combine:
                num_dict[ch][sample] = 0

            # get list of parquet files that have been processed
            parquet_files = glob.glob(f"{idir}/{sample}/outfiles/*_{ch}.parquet")

            if len(parquet_files) == 0:
                continue

            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    print(
                        "Not able to read data: ",
                        parquet_file,
                        " should remove evts from scaling/lumi",
                    )
                    continue
                if len(data) == 0:
                    continue

                # drop AK columns
                for key in data.keys():
                    if data[key].dtype == "object":
                        data.drop(columns=[key], inplace=True)

                if combine:
                    num_dict[ch][single_key] = (
                        num_dict[ch][single_key] + data["tot_weight"].sum()
                    )
                else:
                    num_dict[ch][sample] = (
                        num_dict[ch][sample] + data["tot_weight"].sum()
                    )

        for key, value in num_dict[ch].items():
            if value != 0:
                print(f"number of events for {key} is {value}")

        print(f"-----------------------------------------")

    with open(f"./{odir}/num_dict.pkl", "wb") as f:  # saves counts
        pkl.dump(num_dict, f)


if __name__ == "__main__":
    """
    e.g. run locally as
    python counting_from_pq.py --dir May7_2017 --ch ele --odir counts
    """

    channels = args.ch.split(",")

    year = args.dir[-4:]
    idir = "/eos/uscms/store/user/cmantill/boostedhiggs/" + args.dir

    if year == "2018":
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch

    samples = os.listdir(f"{idir}")

    # make directory to hold counts
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    compute_counts(channels, samples, args.odir, data_label)
