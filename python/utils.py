#!/usr/bin/python

import json
import pickle as pkl
import warnings

import hist as hist2

warnings.filterwarnings("ignore", message="Found duplicate branch ")

# string that matches samples together
add_samples = {
    "Data": ["SingleElectron", "EGamma", "SingleMuon", "JetHT"],
    "QCD": "QCD_Pt",
    "DYJets": "DYJets",
    "WZQQ": "JetsToQQ",
    "SingleTop": "ST",
    "TTbar": "TT",
    "WJetsLNu": "WJetsToLNu",
    "Diboson": ["WW", "WZ", "ZZ"],
    "ttH": ["ttHToNonbb_M125"],
    "WH": ["HWminusJ_HToWW_M-125", "HWplusJ_HToWW_M-125"],
    "ZH": ["HZJ_HToWW_M-125"],
    "ggH": "GluGluHToWW_Pt-200ToInf_M-125",
    "VBF": "VBFHToWWToLNuQQ_M-125_withDipoleRecoil",
}

simplified_labels = {
    "SingleElectron": "Data",
    "EGamma": "Data",
    "SingleMuon": "Data",
    "JetHT": "Data",
    "Data": "Data",
    "QCD": "Multijet",
    "DYJets": r"Z($\ell\ell$) + jets",
    "WJetsLNu": r"W($\ell\nu$) + jets",
    "Diboson": r"VV",
    "WH": r"WH$\rightarrow$WW",
    "ZH": r"ZH$\rightarrow$WW",
    "ggH": r"ggH$\rightarrow$WW",
    "VBF": r"VBF$\rightarrow$WW",
    "ttH": r"ttH$\rightarrow$WW",
    "TTbar": r"$t\bar{t}$+jets",
    "SingleTop": r"single-t",
    "WZQQ": r"W/Z(qq) + jets",
}

# references for signals and data
signal_ref = [
    "ttH",
    "WH",
    "ZH",
    "ggH",
    "VBF",
]
data_ref = ["SingleElectron", "EGamma", "SingleMuon", "JetHT"]

label_by_ch = {
    "ele": "Electron",
    "mu": "Muon",
    "all": "Semi-leptonic",
    "lep": "Semi-leptonic",
}

color_by_sample = {
    "QCD": "tab:orange",
    "DYJets": "tab:purple",
    "WJetsLNu": "tab:green",
    "TTbar": "tab:blue",
    "WZQQ": "salmon",
    "SingleTop": "tab:cyan",
    "Diboson": "orchid",
    "ttH": "tab:olive",
    "ggH": "coral",
    "WH": "tab:brown",
    "ZH": "darkred",
    "VBF": "tab:gray",
}


def get_sample_to_use(sample, year, is_data):
    """
    Get name of sample that adds small subsamples
    """
    single_sample = None
    for single_key, key in add_samples.items():
        if type(key) is list:
            for k in key:
                if k in sample:
                    single_sample = single_key
        else:
            if key in sample:
                single_sample = single_key

    if year == "Run2" and is_data:
        single_sample = "Data"

    if single_sample is not None:
        sample_to_use = single_sample
    else:
        sample_to_use = sample

    # print(f"sample is {sample} - going to use {sample_to_use} in histogram")
    return sample_to_use


def get_sum_sumgenweight(pkl_files, year, sample):
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


def get_cutflow(cut_keys, pkl_files, yr, sample, weight, ch):
    """
    Get cutflow from metadata but multiply by xsec-weight
    """
    evyield = dict.fromkeys(cut_keys, 0)
    for ik, pkl_file in enumerate(pkl_files):
        with open(pkl_file, "rb") as f:
            metadata = pkl.load(f)
            if ch == "lep":
                cutflows = metadata[sample][yr]["cutflows"]
            else:
                cutflows = metadata[sample][yr]["cutflows"][ch]

            for key in cut_keys:
                if key in cutflows.keys():
                    evyield[key] += cutflows[key] * weight
    return evyield


def get_cutflow_axis(cut_keys):
    return hist2.axis.Regular(
        len(cut_keys),
        0,
        len(cut_keys),
        name="var",
        label=r"Event Cutflow",
        overflow=True,
    )


axis_dict = {
    "Zmass": hist2.axis.Regular(40, 30, 450, name="var", label=r"Zmass [GeV]", overflow=True),
    "lep_pt": hist2.axis.Regular(40, 30, 450, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "fj_minus_lep_mass": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "fj_minus_lep_pt": hist2.axis.Regular(40, 0, 450, name="var", label=r"Jet - Lepton $p_T$ [GeV]", overflow=True),
    "lep_met_mt": hist2.axis.Regular(35, 0, 400, name="var", label=r"$m_T(lep, p_T^{miss})$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "lep_fj_dr": hist2.axis.Regular(35, 0.0, 0.8, name="var", label=r"$\Delta R(l, Jet)$", overflow=True),
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 200, 1000, name="var", label=r"Jet $p_T$ [GeV]", overflow=True),
    "fj_msoftdrop": hist2.axis.Regular(45, 20, 400, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
}
