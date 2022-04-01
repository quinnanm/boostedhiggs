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
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def get_simplified_label(sample):   # get simplified "alias" names of the samples for plotting purposes
    f = open('plot_configs/simplified_labels.json')
    name = json.load(f)
    f.close()
    if sample in name.keys():
        return str(name[sample])
    else:
        return sample


def get_sum_sumgenweight(idir, year, sample):
    pkl_files = glob.glob(f'{idir}/{sample}/outfiles/*.pkl')  # get the pkl metadata of the pkl files that were processed
    sum_sumgenweight = 1  # TODO why not 0
    for file in pkl_files:
        # load and sum the sumgenweight of each
        with open(file, 'rb') as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]['sumgenweight']
    return sum_sumgenweight


# define the axes for the different variables to be plotted
# define samples
signal_by_ch = {
    'ele': ['GluGluHToWWToLNuQQ'],
    'mu': ['GluGluHToWWToLNuQQ'],
    'had': ['GluGluHToWWTo4q', 'GluGluHToWWTo4q_ext1'],
}
data_by_ch = {
    'ele': 'SingleElectron',
    'mu': 'SingleMuon',
    'had': 'JetHT',
}

color_by_sample = {
    "QCD": 'tab:orange',
    "DYJets": 'tab:purple',
    "WJetsLNu": 'tab:green',
    "TTbar": 'tab:blue',
    "ZQQ": 'tab:pink',
    "WQQ": 'tab:red',
    "SingleTop": 'tab:gray',
}
# available tab colors
# 'tab:cyan'
# 'tab:olive'
# 'tab:gray'
# 'tab:brown':

add_samples = {
    'SingleElectron': 'SingleElectron',
    'SingleMuon': 'SingleMuon',
    'JetHT': 'JetHT',
    'QCD': 'QCD_Pt',
    'DYJets': 'DYJets',
    'ZQQ': 'ZJetsToQQ',
    'WQQ': 'WJetsToQQ',
    'SingleTop': 'ST',
    'TTbar': 'TT',
    'WJetsLNu': 'WJetsToLNu',
}

axis_dict = {
    'lep_pt': hist2.axis.Regular(50, 20, 500, name='var', label=r'Lepton $p_T$ [GeV]'),
    'lep_isolation': hist2.axis.Regular(20, 0, 3.5, name='var', label=r'Lepton iso'),
    'lep_misolation': hist2.axis.Regular(20, 0, 3.5, name='var', label=r'Lepton mini iso'),
    'lep_fj_m': hist2.axis.Regular(20, 0, 200, name='var', label=r'Jet - Lepton mass [GeV]'),
    'lep_fj_bjets_ophem': hist2.axis.Regular(15, 0, 0.31, name='var', label=r'btagFlavB (opphem)'),
    'lep_fj_dr': hist2.axis.Regular(15, 0, 0.8, name='var', label=r'$\Delta R(l, Jet)$'),
    'mu_mvaId': hist2.axis.Regular(20, -1, 1, name='var', label='Muon MVAID'),
    'fj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet $m_{sd}$ [GeV]'),
    'fj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet $p_T$ [GeV]'),
    'fj0_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet$_0$ $m_{sd}$ [GeV]'),
    'fj0_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet$_0$ $p_T$ [GeV]'),
    'fj1_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet$_1$ $m_{sd}$ [GeV]'),
    'fj1_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet$_1$ $p_T$ [GeV]'),
    'lep_met_mt': hist2.axis.Regular(20, 0, 300, name='var', label=r'$m_T(lep, p_T^{miss})$ [GeV]'),
    'ht':  hist2.axis.Regular(20, 180, 1500, name='var', label='HT [GeV]'),
    'met':  hist2.axis.Regular(50, 0, 400, name='var', label='MET [GeV]'),
    'fj0_bjets_ophem': hist2.axis.Regular(15, 0, 0.4, name='var', label=r'btagFlavB (opphem)'),
    'fj0_pnh4q': hist2.axis.Regular(30, 0, 1, name='var', label=r'Jet$_0$ PNH4q non-MD'),
    'fj1_pnh4q': hist2.axis.Regular(30, 0, 1, name='var', label=r'Jet$_1$ PNH4q non-MD'),
    'lep_matchedH': hist2.axis.Regular(20, 100, 400, name='var', label=r'matchedH $p_T$ [GeV]'),
    'had_matchedH': hist2.axis.Regular(20, 100, 400, name='var', label=r'matchedH $p_T$ [GeV]'),
    'lep_nprongs': hist2.axis.Regular(20, 0, 4, name='var', label=r'num of prongs'),
    'had_nprongs': hist2.axis.Regular(20, 0, 4, name='var', label=r'num of prongs'),
}
# be careful
