#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018
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


def make_2dplots(year, ch, idir, odir, samples, vars, x_bins, x_start, x_end, y_bins, y_start, y_end, make_iso_cuts):
    """
    Makes 2D plots of two variables

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
        vars: a list of two variable names to plot against each other... see the full list of choices in plot_configs/vars.json
    """

    # instantiates the histogram object
    hists = hist2.Hist(
        hist2.axis.Regular(x_bins, x_start, x_end, name=vars[0], label=vars[0], overflow=True),
        hist2.axis.Regular(y_bins, y_start, y_end, name=vars[1], label=vars[1], overflow=True),
        hist2.axis.StrCategory([], name='samples', growth=True),
    )

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

            if vars[1] == 'tagger_score':
                vars[1] = f'{ch}_score'

            x = data[vars[0]][select_var]
            y = data[vars[1]][select_var]

            hists.fill(
                x,
                y,
                sample_to_use,
            )

    print("------------------------------------------------------------")

    with open(f'{odir}/{ch}_{vars[0]}_{vars[1]}.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_2dplots(year, ch, odir, vars):
    """
    Plots 2D plots of two variables that were made by "make_2dplots" function

    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
        vars: a list of two variable names to plot against each other... see the full list of choices in plot_configs/vars.json
    """

    # load the hists
    with open(f'{odir}/{ch}_{vars[0]}_{vars[1]}.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make directory to store stuff per year
    if not os.path.exists(f'{odir}/{ch}_{vars[0]}_{vars[1]}'):
        os.makedirs(f'{odir}/{ch}_{vars[0]}_{vars[1]}')

    # make plots per channel
    for sample in hists.axes[2]:
        print(sample)
        # one for log z-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        hep.hist2dplot(hists[{'samples': sample}], ax=ax, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=10000))
        ax.set_xlabel(f"{vars[0]}")
        ax.set_ylabel(f"{vars[1]}")
        ax.set_title(f'{ch} channel for \n {sample}')
        hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        print(f'saving at {odir}/{ch}_{vars[0]}_{vars[1]}/{sample}_log_z.png')
        plt.savefig(f'{odir}/{ch}_{vars[0]}_{vars[1]}/{sample}_log_z.png')
        plt.close()

        # # one for non-log z-scale
        # fig, ax = plt.subplots(figsize=(8, 5))
        # hep.hist2dplot(hists[{'samples': sample}], ax=ax, cmap="plasma")
        # ax.set_xlabel(f"{vars[0]}")
        # ax.set_ylabel(f"{vars[1]}")
        # ax.set_title(f'{ch} channel for \n {sample}')
        # hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
        # hep.cms.text("Work in Progress", ax=ax)
        # print(f'saving at {odir}/{ch}_{vars[0]}_{vars[1]}/{sample}.pdf')
        # plt.savefig(f'{odir}/{ch}_{vars[0]}_{vars[1]}/{sample}.pdf')
        # plt.close()


def main(args):

    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    channels = args.channels.split(',')

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
    vars = args.vars.split(',')

    for ch in channels:

        if args.make_hists:
            print(f'Making 2dplot of {vars} for {ch} channel')
            make_2dplots(args.year, ch, args.idir, odir, samples, vars, args.x_bins, args.x_start, args.x_end, args.y_bins, args.y_start, args.y_end, args.make_iso_cuts)

        if args.plot_hists:
            print('Plotting...')
            plot_2dplots(args.year, ch, odir, vars)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_iso vs lep_pt:   python make_2dplots.py --year 2017 --odir 2d_plots --channels ele,mu --vars lep_isolation,lep_pt --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 1 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20_2017/ --make_hists
    # lep_pt vs lep_fj_dr: python make_2dplots.py --year 2017 --odir 2d_plots --channels ele --vars lep_pt,lep_fj_dr     --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0.1 --y_end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # fj_pt vs lep_fj_dr:  python make_2dplots.py --year 2017 --odir 2d_plots --channels ele,mu --vars fj_pt,lep_fj_dr      --make_hists --plot_hists --x_bins 100 --x_start 200 --x_end 500 --y_bins 100 --y_start 0.1 --y_end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs mt:        python make_2dplots.py --year 2017 --odir 2d_plots --channels ele --vars lep_pt,lep_met_mt    --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_pt vs fj_pt:     python make_2dplots.py --year 2017 --odir 2d_plots --channels ele --vars lep_pt,fj_pt         --make_hists --plot_hists --x_bins 100 --x_start 0 --x_end 500 --y_bins 100 --y_start 0   --y_end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    # lep_miso vs tagger_score:     python make_2dplots.py --year 2017 --odir 2d_plots --channels ele --vars lep_misolation,tagger_score --make_hists --plot_hists --x_bins 10 --x_start 0 --x_end 2 --y_bins 10 --y_start 0   --y_end 1 --idir /eos/uscms/store/user/cmantill/boostedhiggs/Sep2

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',            dest='year',        default='2017',                                 help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano_value.json",     help="path to json with samples to be plotted")
    parser.add_argument('--channels',        dest='channels',    default='ele',                                  help="channel... choices are ['ele', 'mu', 'had']")
    parser.add_argument('--odir',            dest='odir',        default='hists',                                help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                          help="input directory with results")
    parser.add_argument('--vars',            dest='vars',        default='lep_pt,lep_isolation',                 help="channels for which to plot this variable")
    parser.add_argument('--x_bins',          dest='x_bins',      default=50,                                     help="binning of the first variable passed",                type=int)
    parser.add_argument('--x_start',         dest='x_start',     default=0,                                      help="starting range of the first variable passed",         type=float)
    parser.add_argument('--x_end',           dest='x_end',       default=1,                                      help="end range of the first variable passed",              type=float)
    parser.add_argument('--y_bins',          dest='y_bins',      default=50,                                     help="binning of the second variable passed",               type=int)
    parser.add_argument('--y_start',         dest='y_start',     default=0,                                      help="starting range of the second variable passed",        type=float)
    parser.add_argument('--y_end',           dest='y_end',       default=1,                                      help="end range of the second variable passed",             type=float)
    parser.add_argument("--make_hists",      dest='make_hists',  action='store_true',                            help="Make hists")
    parser.add_argument("--plot_hists",      dest='plot_hists',  action='store_true',                            help="Plot the hists")
    parser.add_argument("--make_iso_cuts",      dest='make_iso_cuts',  action='store_true',     help="Make iso cuts")

    args = parser.parse_args()

    main(args)
