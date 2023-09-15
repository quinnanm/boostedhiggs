#!/usr/bin/python


import warnings
from typing import List

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from hist import Hist

plt.style.use(hep.style.CMS)

warnings.filterwarnings("ignore", message="Found duplicate branch ")

# (name in templates, name in cards)
labels = {
    # sigs
    "ggF": "ggF",
    "VBF": "VBF",
    "VH": "VH",
    "ttH": "ttH",
    # BKGS
    "QCD": "qcd",
    "TTbar": "ttbar",
    "WJetsLNu": "wjets",
    "SingleTop": "singletop",
    "DYJets": "zjets",
}

bkgs = ["TTbar", "WJetsLNu", "SingleTop", "DYJets"]

sigs = ["ggF", "VBF", "VH", "ttH"]
samples = sigs + bkgs


def shape_to_num(var, nom, clip=1.5):
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)

    if abs(var_rate / nom_rate) > clip:
        var_rate = clip * nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate / nom_rate


def get_template(h, sample, ptbin):
    mass_axis = 3
    massbins = h.axes[mass_axis].edges
    return (h[{"samples": sample, "systematic": "nominal", "fj_pt": ptbin}].values(), massbins, "reco_higgs_m")


def blindBins(h: Hist, blind_region: List, blind_samples: List[str] = []):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_samples`` specified, only blind those samples, else blinds all.

    CAREFUL: assumes axis=0 is samples, axis=3 is mass_axis

    """

    h = h.copy()

    #     mass_axis = np.argmax(np.array(list(h.axes.name))=="rec_higgs_m")
    mass_axis = 3
    massbins = h.axes[mass_axis].edges

    lv = int(np.searchsorted(massbins, blind_region[0], "right"))
    rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)

    if blind_samples:
        for blind_sample in blind_samples:
            sample_index = np.argmax(np.array(list(h.axes[0])) == blind_sample)
            h.view(flow=True)[sample_index, :, :, lv:rv] = 0

    else:
        h.view(flow=True)[:, :, :, lv:rv] = 0

    return h
