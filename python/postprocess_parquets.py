#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018
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


def append_correct_weights(idir, samples, year, channels, reprocess=False):
    """
    Updates the processed parquet daraftames by appending the correct scaling factor/weight per event as new column 'tot_weight'

    Args:
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    # Get luminosity of year
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    f.close()
    print(f"Processing samples from year {year} with luminosity {luminosity}")

    if year == "2018":
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch

    # loop over the processed files and fill the histograms
    for ch in channels:
        for sample in samples:
            print("------------------------------------------------------------")
            # check if the sample was processed
            pkl_dir = f"{idir}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)  #
            if not pkl_files:  # skip samples which were not processed
                print("- No processed files found...", pkl_dir, "skipping sample...", sample)
                continue

            # define an isdata bool
            is_data = False

            for key in data_label.values():
                if key in sample:
                    is_data = True

            # retrieve xsections for MC and define xsec_weight=1 for data
            if not is_data:
                # Find xsection
                f = open("../fileset/xsec_pfnano.json")
                xsec = json.load(f)
                f.close()
                try:
                    xsec = eval(str((xsec[sample])))
                except:
                    print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
                    continue

                # Get overall weighting of events.. each event has a genweight... sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
                xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(idir, year.replace("APV", ""), sample)
            else:
                xsec_weight = 1

            # get list of parquet files that have been processed
            parquet_files = glob.glob(f"{idir}/{sample}/outfiles/*_{ch}.parquet")

            if len(parquet_files) != 0:
                print(f"Processing {ch} channel of sample", sample)

            for i, parquet_file in enumerate(parquet_files):
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    print("Not able to read data: ", parquet_file, " should remove evts from scaling/lumi")
                    continue

                if len(data) == 0:
                    continue

                if "tot_weight" in data.columns and not reprocess:
                    print(
                        "Warning: File has already been reprocessed! Add --reprocess to arguments if you want to re-writing tot weight."
                    )
                    continue

                if not is_data:
                    event_weight = data["weight"] * data[f"weight_{ch}"]
                else:
                    event_weight = 1  # for data fill a weight column with ones

                # append an additional column 'tot_weight' to the parquet dataframes
                data["tot_weight"] = xsec_weight * event_weight

                # update parquet file (this line should overwrite the stored dataframe)
                pq.write_table(pa.Table.from_pandas(data), parquet_file)

    print("------------------------------------------------------------")


def main(args):

    channels = args.channels.split(",")

    # build samples
    samples = os.listdir(args.idir)

    append_correct_weights(args.idir, samples, args.year, channels, args.reprocess)


if __name__ == "__main__":
    # e.g. python postprocess_parquets.py --channels ele,mu --year 2017 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/Aug11_2017

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", choices=["2016APV", "2016", "2017", "2018"], help="year", required=True)
    parser.add_argument("--channels", dest="channels", default="ele,mu,had", help="channels for which to plot this variable")
    parser.add_argument("--idir", dest="idir", default="../results/", help="input directory with results")
    parser.add_argument(
        "--reprocess", dest="reprocess", action="store_true", help="force re-processing of parquet file to include weight"
    )

    args = parser.parse_args()

    main(args)
