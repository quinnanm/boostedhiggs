#!/usr/bin/python

import glob
import json
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Found duplicate branch ")


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


axis_dict = {
    "Zmass": hist2.axis.Regular(40, 30, 450, name="var", label=r"Zmass [GeV]", overflow=True),
    "lep_pt": hist2.axis.Regular(40, 30, 450, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "fj_minus_lep_m": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "fj_minus_lep_pt": hist2.axis.Regular(40, 0, 450, name="var", label=r"Jet - Lepton $p_T$ [GeV]", overflow=True),
    "lep_met_mt": hist2.axis.Regular(35, 0, 400, name="var", label=r"$m_T(lep, p_T^{miss})$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "lep_fj_dr": hist2.axis.Regular(35, 0.0, 0.8, name="var", label=r"$\Delta R(Jet, Lepton)$", overflow=True),
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 200, 1000, name="var", label=r"Jet $p_T$ [GeV]", overflow=True),
    "fj_msoftdrop": hist2.axis.Regular(35, 20, 250, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
    "rec_higgs_m": hist2.axis.Regular(35, 0, 480, name="var", label=r"Higgs reconstructed mass [GeV]", overflow=True),
    "rec_higgs_pt": hist2.axis.Regular(30, 0, 1000, name="var", label=r"Higgs reconstructed $p_T$ [GeV]", overflow=True),
    "fj_pt_over_lep_pt": hist2.axis.Regular(35, 1, 10, name="var", label=r"$p_T$(Jet) / $p_T$(Lepton)", overflow=True),
    "golden_var": hist2.axis.Regular(35, 0, 10, name="var", label=r"$p_{T}(W_{l\nu})$ / $p_{T}(W_{qq})$", overflow=True),
    "rec_dphi_WW": hist2.axis.Regular(
        35, 0, 3, name="var", label=r"$\left| \Delta \phi(W_{l\nu}, W_{qq}) \right|$", overflow=True
    ),
    "fj_ParT_mass": hist2.axis.Regular(35, 0, 250, name="var", label=r"ParT regressed mass [GeV]", overflow=True),
    "fj_ParticleNet_mass": hist2.axis.Regular(
        35, 0, 250, name="var", label=r"fj_ParticleNet regressed mass [GeV]", overflow=True
    ),
}


# new stuff
combine_samples = {
    # data
    "SingleElectron_": "SingleElectron",
    "SingleMuon_": "SingleMuon",
    "EGamma_": "EGamma",
    # signal
    "GluGluHToWW_Pt-200ToInf_M-125": "HWW",
    "HToWW_M-125": "VH",
    "VBFHToWWToLNuQQ_M-125_withDipoleRecoil": "VBF",
    "ttHToNonbb_M125": "ttH",
    # bkg
    "QCD_Pt": "QCD",
    "DYJets": "DYJets",
    "WJetsToLNu_": "WJetsLNu",
    "JetsToQQ": "WZQQ",
    "TT": "TTbar",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
}
signals = ["HWW", "ttH", "VH", "VBF"]

data_by_ch = {
    "ele": "SingleElectron",
    "mu": "SingleMuon",
}

weights = {
    "mu": {
        "weight_genweight": 1,
        "weight_L1Prefiring": 1,
        "weight_pileup": 1,
        "weight_trigger_iso_muon": 1,
        "weight_trigger_noniso_muon": 1,
        "weight_isolation_muon": 1,
        "weight_id_muon": 1,
        "weight_vjets_nominal": 1,
    },
    "ele": {
        "weight_genweight": 1,
        "weight_L1Prefiring": 1,
        "weight_pileup": 1,
        "weight_trigger_electron": 1,
        "weight_reco_electron": 1,
        "weight_id_electron": 1,
        "weight_vjets_nominal": 1,
    },
}


# tagger definitions
def disc_score(df, sigs, bkgs):

    num = df[sigs].sum(axis=1)
    den = df[sigs].sum(axis=1) + df[bkgs].sum(axis=1)
    return num / den


# scores definition
hwwev = ["fj_PN_probHWqqWev0c", "fj_PN_probHWqqWev1c", "fj_PN_probHWqqWtauev0c", "fj_PN_probHWqqWtauev1c"]
hwwmv = ["fj_PN_probHWqqWmv0c", "fj_PN_probHWqqWmv1c", "fj_PN_probHWqqWtaumv0c", "fj_PN_probHWqqWtaumv1c"]
qcd = ["fj_PN_probQCDbb", "fj_PN_probQCDcc", "fj_PN_probQCDb", "fj_PN_probQCDc", "fj_PN_probQCDothers"]

tope = ["fj_PN_probTopbWev", "fj_PN_probTopbWtauev"]
topm = ["fj_PN_probTopbWmv", "fj_PN_probTopbWtaumv"]
tophad = ["fj_PN_probTopbWqq0c", "fj_PN_probTopbWqq1c", "fj_PN_probTopbWq0c", "fj_PN_probTopbWq1c", "fj_PN_probTopbWtauhv"]

top = tope + topm + tophad

sigs = {
    "ele": hwwev,
    "mu": hwwmv,
}

qcd_bkg = [b.replace("PN", "ParT") for b in qcd]
top_bkg = [b.replace("PN", "ParT") for b in tope + topm + tophad]
inclusive_bkg = [b.replace("PN", "ParT") for b in qcd + tope + topm + tophad]


def event_skimmer(
    channels,
    samples_dir,
    samples,
    presel,
    columns="all",
    add_inclusive_score=False,
    add_qcd_score=False,
    add_top_score=False,
):

    events_dict = {}
    for ch in channels:
        events_dict[ch] = {}

        # for the tagger
        new_sig = [s.replace("PN", "ParT") for s in sigs[ch]]

        # get lumi
        with open("../fileset/luminosity.json") as f:
            luminosity = json.load(f)[ch]["2017"]

        condor_dir = os.listdir(samples_dir)
        for sample in condor_dir:

            # get a combined label to combine samples of the same process
            for key in combine_samples:
                if key in sample:
                    sample_to_use = combine_samples[key]
                    break
                else:
                    sample_to_use = sample

            if sample_to_use not in samples:
                continue

            is_data = False
            if sample_to_use == data_by_ch[ch]:
                is_data = True

            print(f"Finding {sample} samples and should combine them under {sample_to_use}")

            out_files = f"{samples_dir}/{sample}/outfiles/"
            parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            pkl_files = glob.glob(f"{out_files}/*.pkl")

            if not parquet_files:
                print(f"No parquet file for {sample}")
                continue

            data = pd.read_parquet(parquet_files)
            if len(data) == 0:
                continue

            # replace the weight_pileup of the strange events with the mean weight_pileup of all the other events
            strange_events = data["weight_pileup"] > 6
            if len(strange_events) > 0:
                data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

            # apply selection
            print("---> Applying preselection.")
            for selection in presel[ch]:
                print(f"applying {selection} selection on {len(data)} events")
                data = data.query(presel[ch][selection])
            print("---> Done with preselection.")

            # get event_weight
            if not is_data:
                print("---> Accumulating event weights.")
                event_weight = get_xsecweight(pkl_files, "2017", sample, is_data, luminosity)
                for w in weights[ch]:
                    if w not in data.keys():
                        print(f"{w} weight is not stored in parquet")
                        continue
                    event_weight *= data[w]
                print("---> Done with accumulating event weights.")
            else:
                event_weight = np.ones_like(data["fj_pt"])

            data["event_weight"] = event_weight

            # add tagger scores
            if add_qcd_score:
                data["QCD_score"] = disc_score(data, new_sig, qcd_bkg)
            if add_top_score:
                data["Top_score"] = disc_score(data, new_sig, top_bkg)
            if add_inclusive_score:
                data["inclusive_score"] = disc_score(data, new_sig, inclusive_bkg)

            print(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
            print(f"tot event weight {data['event_weight'].sum()} \n")

            if columns == "all":
                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data])
            else:
                # specify columns to keep
                cols = columns + ["event_weight"]
                if add_qcd_score:
                    cols += ["QCD_score"]
                if add_top_score:
                    cols += ["Top_score"]
                if add_inclusive_score:
                    cols += ["inclusive_score"]

                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data[cols]
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data[cols]])

    return events_dict
