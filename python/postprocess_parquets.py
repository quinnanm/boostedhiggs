#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
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


def append_correct_weights(idir, samples, years, channels):
    """
    Updates the processed parquet daraftames by appending the correct scaling factor/weight per event as new column 'tot_weight'

    Args:
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                print("------------------------------------------------------------")
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of {sample}')
                else:
                    print(f'No processed files for {sample} are found')

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue
                    if len(data) == 0:
                        continue

                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)... these will be avoided when we add fj_pt>200 cut for leptonic channels
                    if ch != 'had':
                        data = data[data['fj_pt'] != -1]

                    if ((data["leptonInJet"] != 1).sum() != 0):

                        print(data[data["leptonInJet"] != 1]["leptonInJet"])
                    # try:
                    #     event_weight = data['weight'].to_numpy()
                    #     # Find xsection if MC
                    #     f = open('../fileset/xsec_pfnano.json')
                    #     xsec = json.load(f)
                    #     f.close()
                    #     xsec = eval(str((xsec[sample])))
                    #
                    #     # Get overall weighting of events
                    #     xsec_weight = (xsec * luminosity[year]) / (get_sum_sumgenweight(idir, year, sample))
                    #
                    # except:  # for data
                    #     data['weight'] = 1  # for data fill a weight column with ones
                    #     xsec_weight = 1
                    #
                    # # append an additional column 'tot_weight' to the parquet dataframes
                    # data['tot_weight'] = xsec_weight * data['weight']
                    #
                    # # update parquet file (this line should overwrite the stored dataframe)
                    # pq.write_table(pa.Table.from_pandas(data), parquet_file)

    print("------------------------------------------------------------")


def main(args):

    years = args.years.split(',')
    channels = args.channels.split(',')

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    append_correct_weights(args.idir, samples, years, channels)


if __name__ == "__main__":
    # e.g. run locally as
    # python postprocess_parquets.py --year 2017 --channels had --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                                 help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json",     help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                           help='channels for which to plot this variable')
    parser.add_argument('--idir',            dest='idir',        default='../results/',                          help="input directory with results")

    args = parser.parse_args()

    main(args)
