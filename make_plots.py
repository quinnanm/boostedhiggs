#!/usr/bin/python

import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def main(args):

    # load result of 2 jobs
    samples = args.sample.split(',')
    year = args.year
    # num_jobs = args.endi - args.starti

    channels = ['ele', 'mu', 'had']

    for sample in samples:
        print('Processing sample', sample)
        sum_sumgenweight = {}
        sum_sumgenweight[sample] = 0
        data_all = {}
        pkl_files = glob.glob(f'./results/{sample}/outfiles/*.pkl')  # get list of metadata pkl files that need to be processed
        if not pkl_files:  # skip samples which were not processed
            continue
        print('lol')

        for ch in channels:
            print('Processing channel', ch)
            parquet_files = glob.glob(f'./results/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that need to be processed
            for i, parquet_file in enumerate(parquet_files):
                tmp = pq.read_table(parquet_file).to_pandas()
                if i == 0:
                    data = tmp
                else:
                    data = pd.concat([data, tmp], ignore_index=True)

                # load and sum the sumgenweight of each
                with open(pkl_files[i], 'rb') as f:
                    metadata = pkl.load(f)
                sum_sumgenweight[sample] = sum_sumgenweight[sample] + metadata[sample][year]['sumgenweight']

            print('# of files processed is', i + 1)
            data_all[ch] = data

        xsec = {}
        xsec[sample] = 2

        luminosity = {}
        luminosity[year] = 3

        xsec_weight = {}
        xsec_weight[sample] = (xsec[sample] * luminosity[year]) / (sum_sumgenweight[sample])

        # for leptonic channel
        leppt = {}
        event_weight = {}
        for ch in ['ele', 'mu']:
            leppt[ch] = data_all[ch]['lepton_pt'].to_numpy()
            event_weight[ch] = data_all[ch]['weight'].to_numpy()

        # now we can make histograms for higgspt, jetpt, leptonpt
        import hist as hist2
        print('Making histograms...')
        channel_cat = hist2.axis.StrCategory([], name='channel', growth=True)

        leppt_axis = hist2.axis.Regular(25, 10, 400, name='leppt', label=r'Lepton $p_T$ [GeV]')

        hists = hist2.Hist(
            channel_cat,
            leppt_axis,
        )

        hists.fill(
            channel="ele",
            leppt=leppt['ele'],
            weight=event_weight['ele'] * xsec_weight[sample],
            # weight=xsec_weight[sample],
        )
        hists.fill(
            channel="mu",
            leppt=leppt['mu'],
            weight=event_weight['mu'] * xsec_weight[sample],
            # weight=xsec_weight[sample],
        )

        # now we plot trigger efficiency as function of jetpt
        import matplotlib.pyplot as plt
        import mplhep as hep
        from hist.intervals import clopper_pearson_interval

        def get_yerr(num, den):
            return abs(clopper_pearson_interval(num.view(), den.view()) - num.view() / den.view())

        fig, ax = plt.subplots(1, 1)
        hep.histplot(hists[{"channel": "ele"}],
                     # yerr=get_yerr(num_nom),
                     ax=ax,
                     # histtype='errorbar', color='red', capsize=4, elinewidth=1,
                     label="electron channel",
                     # density=True
                     )
        hep.histplot(hists[{"channel": "mu"}],
                     # yerr=get_yerr(num_nom),
                     ax=ax,
                     # histtype='errorbar', color='red', capsize=4, elinewidth=1,
                     label="muon channel",
                     # density=True
                     )
        # ax.set_ylim(0,1)
        ax.set_title(f'{sample}')
        ax.legend()

        if not os.path.exists('hists'):
            os.makedirs('hists')

        plt.savefig(f'hists/{sample}.pdf')


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_plots.py --year 2017 --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8,QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    # parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    # parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument('--sample',     dest='sample',     default=None,         help='sample name', required=True)
    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    args = parser.parse_args()

    main(args)
