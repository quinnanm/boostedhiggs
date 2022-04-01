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


def make_1dhist_ratio(idir, odir, samples, years, channels, var1, var2, bins, range, cut=None):
    '''
    makes and plots 1d histograms of a ratio of two variables
    - var1: numerator
    - var2: denominator
    '''

    hists = {}
    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}
        for ch in channels:  # initialize the histograms for the different channels and different variables
            hists[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, range[0], range[1], name=var1 + '/' + var2, label=var1 + '/' + var2, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)     # to combine different pt bins of the same process
            )

        # make directory to store stuff per year
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}/')
        if not os.path.exists(f'{odir}/plots_{year}/{var1}_{var2}'):
            os.makedirs(f'{odir}/plots_{year}/{var1}_{var2}')

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample', sample)
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

                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    data = data[data[var1] != -1]
                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    data = data[data[var2] != -1]

                    if cut == "btag":
                        data = data[data["anti_bjettag"] == 1]
                        cut = 'preselection + btag'
                    elif cut == "dr":
                        data = data[data["leptonInJet"] == 1]
                        cut = 'preselection + leptonInJet'
                    elif cut == "btagdr":
                        data = data[data["anti_bjettag"] == 1]
                        data = data[data["leptonInJet"] == 1]
                        cut = 'preselection + btag + leptonInJet'
                    else:
                        cut = 'preselection'

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        hists[year][ch].fill(
                            data[var1] / data[var2], single_sample,  # combining all events under one name
                        )
                    else:
                        hists[year][ch].fill(
                            data[var1] / data[var2], sample,
                        )
                print(f"Applied {cut} cuts")

            for sample in hists[year][ch].axes[-1]:

                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample}], ax=ax)
                ax.set_xlabel(f"{var1}/{var2}")
                ax.set_title(f'{ch} channel for \n {sample} \n with {cut} cut')
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var1}_{var2}/{ch}_{sample}_{cut}.pdf')
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

    range = [args.start, args.end]
    print(f'Plotting {vars[0]}/{vars[1]} histogram')
    make_1dhist_ratio(args.idir, args.odir, samples, years, channels, vars[0], vars[1], args.bins, range, args.cut)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt/fj_pt: python make_1dhist_ratio.py --year 2017 --odir hists/1dhists_ratio --channels ele --vars lep_pt,fj_pt --bins 100 --start 0 --end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists/1dhists_ratio',              help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument('--vars',            dest='vars',        default='lep_pt,fj_pt',                     help='variables to plot... specefy it in the form num,den')
    parser.add_argument('--bins',            dest='bins',        default=50,                                 help="binning of the first variable passed",                type=int)
    parser.add_argument('--start',           dest='start',       default=0,                                  help="starting range of the first variable passed",         type=int)
    parser.add_argument('--end',             dest='end',         default=1,                                  help="end range of the first variable passed",              type=int)
    parser.add_argument('--cut',             dest='cut',         default=None,                               help="specify cut... choices are ['btag', 'dr', 'btagdr'] otherwise only preselection is applied")

    args = parser.parse_args()

    main(args)
