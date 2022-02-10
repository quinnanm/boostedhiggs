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

import hist as hist2
import matplotlib.pyplot as plt
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def get_simplified_label(sample):
    f = open('../data/simplified_labels.json')
    name = json.load(f)
    f.close()
    return name[sample]


def make_unstacked_hists(variable_name, channels, samples, data_all, xsec_weight, year):
    print(f'- Making histogram of {variable_name}')
    if variable_name == 'lepton_pt':
        if 'had' in channels:
            channels.remove('had')
            print(f'skipping hadronic channel to make {variable_name} histogram')

        for ch in channels:
            leppt = {}
            event_weight = {}

            sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)
            leppt_axis = hist2.axis.Regular(25, 10, 400, name='leppt', label=r'Lepton $p_T$ [GeV]')

            hists = hist2.Hist(
                sample_axis,
                leppt_axis,
            )

            fig, ax = plt.subplots(1, 1)

            for sample in samples:
                print(sample)
                leppt[sample] = data_all[sample][ch]['lepton_pt'].to_numpy()
                event_weight[sample] = data_all[sample][ch]['weight'].to_numpy()

                hists.fill(
                    samples=sample,
                    leppt=leppt[sample],
                    weight=event_weight[sample] * xsec_weight[sample],
                )

                def get_yerr(num, den):
                    return abs(clopper_pearson_interval(num.view(), den.view()) - num.view() / den.view())

                hep.histplot(hists[{"samples": sample}],
                             # yerr=get_yerr(num_nom),
                             ax=ax,
                             # histtype='errorbar', color='red', capsize=4, elinewidth=1,
                             label=get_simplified_label(sample),
                             # density=True
                             )
            # ax.set_ylim(0,1)
            ax.set_title(f'{ch} channel')
            ax.legend()

            plt.savefig(f'hists/hists_{year}/{variable_name}_{ch}.pdf')
