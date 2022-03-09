#!/usr/bin/python

from axes import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
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


def make_2dplot(idir, odir, samples, years, channels, vars, x_bins=50, x_start=0, x_end=1, y_bins=50, y_start=0, y_end=1):
    # for readability
    x = vars[0]
    y = vars[1]

    hists = {}
    for year in years:
        hists[year] = {}

        for ch in channels:  # initialize the histograms for the different channels and different variables

            hists[year][ch] = hist2.Hist(
                hist2.axis.Regular(x_bins, x_start, x_end, name=x, label=x, flow=False),
                hist2.axis.Regular(y_bins, y_start, y_end, name=y, label=y, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)
            )

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample', sample)

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue
                    if len(data) == 0:
                        continue

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        hists[year][ch].fill(
                            data[x], data[y], single_sample,  # combining all events under one name
                        )
                    else:
                        hists[year][ch].fill(
                            data[x], data[y], sample,
                        )

            for sample in hists[year][ch].axes[-1]:

                fig, ax = plt.subplots(figsize=(8, 5))
                hep.hist2dplot(hists[year][ch][{'samples': sample}], ax=ax, norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="plasma")
                ax.set_xlabel(f"{x}")
                ax.set_ylabel(f"{y}")
                ax.set_title(f'{ch} channel for \n {sample}')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)

                if not os.path.exists(f'{odir}/plots_{year}/'):
                    os.makedirs(f'{odir}/plots_{year}/')
                if not os.path.exists(f'{odir}/plots_{year}/{x}_vs_{y}'):
                    os.makedirs(f'{odir}/plots_{year}/{x}_vs_{y}')

                plt.savefig(f'{odir}/plots_{year}/{x}_vs_{y}/{ch}_{sample}.pdf')
                plt.close()


def main(args):
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    years = args.years.split(',')
    channels = args.channels.split(',')
    vars = args.vars.split(',')

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

    print(f'The 2 variables for cross check are: {vars}')
    make_2dplot(args.idir, args.odir, samples, years, channels, vars, args.x_bins, args.x_start, args.x_end, args.y_bins, args.y_start, args.y_end)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt vs lep_iso: python make_2dplots.py --year 2017 --odir 2dplots --samples configs/samples_pfnano.json --channels ele,mu --vars lep_pt,lep_isolation --x_bins 50 --x_start 0 --x_end 500 --y_bins 50 --y_start 0 --y_end 1 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs
    # lep_pt vs dR: python make_2dplots.py --year 2017 --odir 2dplots --samples configs/samples_pfnano.json --channels ele,mu --vars lep_pt,lep_fj_dr --x_bins 50 --x_start 0 --x_end 500 --y_bins 50 --y_start 0 --y_end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs
    # lep_pt vs mt: python make_2dplots.py --year 2017 --odir 2dplots --samples configs/samples_pfnano.json --channels ele,mu --vars lep_pt,lep_met_mt --x_bins 50 --x_start 0 --x_end 500 --y_bins 50 --y_start 0 --y_end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                        help="year")
    parser.add_argument('--samples',         dest='samples',     default="configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                  help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='2dplots',                     help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                 help="input directory with results")
    parser.add_argument('--vars',            dest='vars',        default='lep_pt,lep_isolation',        help='channels for which to plot this variable')
    parser.add_argument('--x_bins',          dest='x_bins',      default=50,                            help="binning of the first variable passed",                type=int)
    parser.add_argument('--x_start',         dest='x_start',     default=0,                             help="starting range of the first variable passed",         type=int)
    parser.add_argument('--x_end',           dest='x_end',       default=1,                             help="end range of the first variable passed",              type=int)
    parser.add_argument('--y_bins',          dest='y_bins',      default=50,                            help="binning of the second variable passed",               type=int)
    parser.add_argument('--y_start',         dest='y_start',     default=0,                             help="starting range of the second variable passed",        type=int)
    parser.add_argument('--y_end',           dest='y_end',       default=1,                             help="end range of the second variable passed",             type=int)

    args = parser.parse_args()

    main(args)
