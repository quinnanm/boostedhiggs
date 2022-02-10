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


def make_stacked_hists(variable_name, bin_width, low, high, channels, samples, data_all, xsec_weight, year, axis_label):
    print(f'- Making histogram of {variable_name}')

    for ch in channels:
        var = {}
        event_weight = {}

        fig, ax = plt.subplots(1, 1)

        sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)
        var_axis = hist2.axis.Regular(bin_width, low, high, name='var', label=axis_label)

        hists = hist2.Hist(
            sample_axis,
            var_axis,
        )

        hist_samples = []
        labels = []
        signal = 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8'

        for i, sample in enumerate(samples):
            var[sample] = data_all[sample][ch][variable_name].to_numpy()
            event_weight[sample] = data_all[sample][ch]['weight'].to_numpy()

            if sample == signal:  # keep the signal seperate from the other "background" samples
                hists.fill(
                    samples=sample,
                    var=var[sample],
                    weight=event_weight[sample] * xsec_weight[sample],
                )

            elif "QCD" in sample:
                hists.fill(
                    samples="QCD",  # combining all QCD events under one name "QCD"
                    var=var[sample],
                    weight=event_weight[sample] * xsec_weight[sample],
                )
                if "QCD" not in labels:
                    labels.append("QCD")
                    hist_samples.append(hists[{"samples": "QCD"}])

            elif "WJetsToLNu" in sample:  # combining all WJetsToLNu events under one name "WJetsToLNu"
                hists.fill(
                    samples="WJetsToLNu",
                    var=var[sample],
                    weight=event_weight[sample] * xsec_weight[sample],
                )
                if "WJetsToLNu" not in labels:
                    hist_samples.append(hists[{"samples": "WJetsToLNu"}])
                    labels.append("WJetsToLNu")

            else:
                hists.fill(
                    samples=sample,
                    var=var[sample],
                    weight=event_weight[sample] * xsec_weight[sample],
                )
                hist_samples.append(hists[{"samples": sample}])
                labels.append(get_simplified_label(sample))

        # plot the background stacked
        hep.histplot(hist_samples,
                     # yerr=get_yerr(num_nom),
                     ax=ax,
                     stack=True,
                     histtype="fill",
                     label=labels,
                     # density=True
                     )
        # plot the signal seperately on the same plot
        hep.histplot(hists[{"samples": signal}],
                     # yerr=get_yerr(num_nom),
                     ax=ax,
                     stack=True,
                     label="GluGluHToWW",
                     color='red'
                     # density=True
                     )

        ax.set_yscale('log')
        ax.set_title(f'{ch} channel')
        ax.legend()

        hep.cms.lumitext("2017 (13 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        plt.savefig(f'hists/hists_{year}/{variable_name}_{ch}.pdf')

        with open(f'hists/hists_{year}/hists.pkl', 'wb') as f:  # saves a variable that contains the xsec weight of the sample
            pkl.dump(hists, f)
