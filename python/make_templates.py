#!/usr/bin/python

import glob
import json
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
from utils import get_xsecweight

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

# new stuff
combine_samples = {
    # data
    # "SingleElectron_": "SingleElectron",
    "SingleElectron_": "Data",
    # "SingleMuon_": "SingleMuon_",
    "SingleMuon_": "Data",
    # "EGamma_": "EGamma",
    "EGamma_": "Data",
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
    "GluGluHToTauTau": "HTauTau",
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
hwwhad = [
    "fj_PN_probHWqqWqq0c",
    "fj_PN_probHWqqWqq1c",
    "fj_PN_probHWqqWqq2c",
    "fj_PN_probHWqqWq0c",
    "fj_PN_probHWqqWq1c",
    "fj_PN_probHWqqWq2c",
    "fj_PN_probHWqqWtauhv0c",
    "fj_PN_probHWqqWtauhv1c",
]
sigs = hwwev + hwwmv + hwwhad
new_sig = [s.replace("PN", "ParT") for s in sigs]

qcd = ["fj_PN_probQCDbb", "fj_PN_probQCDcc", "fj_PN_probQCDb", "fj_PN_probQCDc", "fj_PN_probQCDothers"]
qcd_bkg = [b.replace("PN", "ParT") for b in qcd]

tope = ["fj_PN_probTopbWev", "fj_PN_probTopbWtauev"]
topm = ["fj_PN_probTopbWmv", "fj_PN_probTopbWtaumv"]
tophad = ["fj_PN_probTopbWqq0c", "fj_PN_probTopbWqq1c", "fj_PN_probTopbWq0c", "fj_PN_probTopbWq1c", "fj_PN_probTopbWtauhv"]
top = tope + topm + tophad
top_bkg = [b.replace("PN", "ParT") for b in tope + topm + tophad]

inclusive_bkg = [b.replace("PN", "ParT") for b in qcd + tope + topm + tophad]


def make_templates(
    year,
    channels,
    samples_dir,
    samples,
    presel,
):
    regions_selections = {
        "cat1_sr": "( (inclusive_score>0.99) & (n_bjets_M < 2) & (lep_fj_dr<0.3) )",
        "wjets_cr": "( (inclusive_score>0.99) & (n_bjets_M < 1) & (lep_fj_dr>0.3) )",
        "tt_cr": "( (inclusive_score<0.90) & (n_bjets_M >=2 ) & (lep_fj_dr>0.3) )",
    }

    # initialzie th histograms
    regions = ["cat1_sr", "wjets_cr", "tt_cr"]
    hists = {}
    for region in regions:
        hists[region] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            hist2.axis.Regular(30, 200, 600, name="fj_pt", label=r"Jet $p_T$ [GeV]", overflow=True),
            hist2.axis.Regular(35, 0, 480, name="rec_higgs_m", label=r"Higgs reconstructed mass [GeV]", overflow=True),
        )

    for ch in channels:
        # get lumi
        luminosity = 0
        with open("../fileset/luminosity.json") as f:
            luminosity += json.load(f)[ch][year]

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
                print(f"ATTENTION: {sample} will be skipped")
                continue

            is_data = False
            if sample_to_use == "Data":
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
            if not is_data:
                strange_events = data["weight_pileup"] > 6
                if len(strange_events) > 0:
                    data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

            # apply selection
            #         print("---> Applying preselection.")
            for selection in presel[ch]:
                #             print(f"applying {selection} selection on {len(data)} events")
                data = data.query(presel[ch][selection])
            #             print("---> Done with preselection.")

            # get event_weight
            if not is_data:
                #                 print("---> Accumulating event weights.")
                event_weight = get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                for w in weights[ch]:
                    if w not in data.keys():
                        #                     print(f"{w} weight is not stored in parquet")
                        continue
                    if weights[ch][w] == 1:
                        #                     print(f"Applying {w} weight")
                        event_weight *= data[w]

            #                 print("---> Done with accumulating event weights.")
            else:
                event_weight = np.ones_like(data["fj_pt"])

            data["event_weight"] = event_weight

            # add tagger scores
            data["inclusive_score"] = disc_score(data, new_sig, inclusive_bkg)

            for region in regions:
                data1 = data.copy()  # get fresh copy of the data to apply selections on
                #             print(f"{region}: applying selection on {len(data1)} events")
                data1 = data1.query(regions_selections[region])
                #             print(f"will fill the {sample_to_use} dataframe with the remaining {len(data1)} events")

                hists[region].fill(
                    samples=sample_to_use,
                    fj_pt=data1["fj_pt"],
                    rec_higgs_m=data1["rec_higgs_m"],
                    weight=data1["event_weight"],
                )
    return hists


if __name__ == "__main__":
    year = "2017"
    channels = ["mu", "ele"]

    samples_dir = f"../Apr12_presel_{year}"

    samples = ["HWW", "VH", "VBF", "ttH", "QCD", "DYJets", "WJetsLNu", "WZQQ", "TTbar", "SingleTop", "Diboson", "Data"]

    presel = {
        "mu": {
            "lep_fj_dr": "( ( lep_fj_dr>0.03) )",
            "fj_pt": "( (fj_pt>220) )",
        },
        "ele": {
            "lep_fj_dr": "( ( lep_fj_dr>0.03) )",
            "fj_pt": "( (fj_pt>220) )",
        },
    }

    hists = make_templates(year, channels, samples_dir, samples, presel)

    tag = "test"
    filehandler = open(f"hists_{tag}.pkl", "wb")
    pkl.dump(hists, filehandler)
