#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
from utils import get_sample_to_use, get_simplified_label, get_sum_sumgenweight
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


def make_big_dataframe(year, ch, idir, odir, samples, vars_to_plot, make_iso_cuts, tag):
    """
    Counts signal and background at different working points of a cut

    Args:
        year: string that represents the year the processed samples are from
        ch: channel to use
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    # Define cuts to make later
    pt_iso = {"ele": 120, "mu": 55}

    # Get luminosity of year
    f = open("../fileset/luminosity.json")
    luminosity = json.load(f)[year]
    f.close()
    print(f"Processing samples from year {year} with luminosity {luminosity} for channel {ch}")

    if year == "2018":
        data_label = data_by_ch_2018
    else:
        data_label = data_by_ch
    
    c = 0
    # loop over the samples
    for sample in samples[year][ch]:
        print(sample)
        # check if the sample was processed
        pkl_dir = f"{idir}_{year}/{sample}/outfiles/*.pkl"
        pkl_files = glob.glob(pkl_dir)
        if not pkl_files:  # skip samples which were not processed
            print("- No processed files found...", pkl_dir, "skipping sample...", sample)
            continue

        # get list of parquet files that have been post processed
        parquet_files = glob.glob(f"{idir}_{year}/{sample}/outfiles/*_{ch}.parquet")

        # define an is_data boolean
        is_data = False
        for key in data_label.values():
            if key in sample:
                is_data = True

        # get xsec weight
        from postprocess_parquets import get_xsecweight
        xsec_weight = get_xsecweight(f"{idir}_{year}", year, sample, is_data, luminosity)

        # get combined sample
        sample_to_use = get_sample_to_use(sample, year)

        for i, parquet_file in enumerate(parquet_files):
            try:
                data = pq.read_table(parquet_file).to_pandas()
            except:
                continue

            try:
                event_weight = data['tot_weight']
            except:
                continue

            if ch == "mu":
                data['mu_score'] = data['fj_isHVV_munuqq'] / \
                    (data['fj_isHVV_munuqq'] + data['fj_ttbar_bmerged'] +
                        data['fj_ttbar_bsplit'] + data['fj_wjets_label'])
            elif ch == "ele":
                data['ele_score'] = data['fj_isHVV_elenuqq'] / \
                    (data['fj_isHVV_elenuqq'] + data['fj_ttbar_bmerged'] +
                        data['fj_ttbar_bsplit'] + data['fj_wjets_label'])

            # make kinematic cuts
            pt_cut = (data["fj_pt"] > 400) & (data["fj_pt"] < 600)
            msd_cut = (data["fj_msoftdrop"] > 30) & (data["fj_msoftdrop"] < 150)

            if make_iso_cuts:
                # make isolation cuts
                iso_cut = (
                    ((data["lep_isolation"] < 0.15) & (data["lep_pt"] < pt_iso[ch])) |
                    (data["lep_pt"] > pt_iso[ch])
                )

                # make mini-isolation cuts
                if ch == "mu":
                    miso_cut = (
                        ((data["lep_misolation"] < 0.1) & (data["lep_pt"] >= pt_iso[ch])) |
                        (data["lep_pt"] < pt_iso[ch])
                    )
                elif ch == "ele":
                    miso_cut = data["lep_pt"] > 10
                    
                select_var = iso_cut & miso_cut
                
            else:
                select_var = (event_weight!=3.14)   # pick all events
                
            if c == 0:  # just so that the first iteration the dataframe is initialized (then for further iterations we can just concat)
                data_all = pd.DataFrame(data[f'{ch}_score'][select_var])
                data_all['sample'] = sample_to_use
                data_all['weight'] = event_weight[select_var]

                for var in vars_to_plot:
                    data_all[var] = data[var][select_var]
                c = c + 1
            else:
                data2 = pd.DataFrame(data[f'{ch}_score'][select_var])
                data2['sample'] = sample_to_use
                data2['weight'] = event_weight[select_var]
                
                for var in vars_to_plot:
                    data2[var] = data[var][select_var]

                data_all = pd.concat([data_all, data2])

        print("------------------------------------------------------------")

    data_all.to_csv(f'{odir}/data_{ch}_{tag}.csv')


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    channels = args.channels.split(',')
    
    # get variables to plot
    f = open(args.vars)
    variables = json.load(f)
    f.close()
    
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

        vars_to_plot = []
        for key, value in variables[ch].items():
            if value == 1:
                vars_to_plot.append(key)

        make_big_dataframe(args.year, ch, args.idir, odir, samples, vars_to_plot, args.make_iso_cuts, args.tag)


if __name__ == "__main__":
    # e.g. run locally as
    # python get_var_for_plotting.py --year 2017 --odir pandas --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2 --make_iso_cuts --tag iso

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument(
        "--vars", dest="vars", default="plot_configs/vars.json", help="path to json with variables to be plotted"
    )    
    parser.add_argument('--odir',            dest='odir',        default='hists',                            help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument('--tag',             dest='tag',        default='',                      help="input directory with results")
    parser.add_argument("--make_iso_cuts",      dest='make_iso_cuts',  action='store_true',     help="Make iso cuts")

    args = parser.parse_args()

    main(args)
