"""
Computes the LateX cutflow tables in the AN.

It does so in a few steps,
    1. Will load the pkl files that contain the cutflows and the sumgenweight
    2. Will scale the events by the cross section
    3. Will save the yields in a dictionary called `cutflows -> Dict()`
    4. Will make the LateX table using the function `make_composition_table()`

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

sys.path
sys.path.append("../python/")

import utils

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

pd.options.mode.chained_assignment = None


cuts = {
    "mu": [
        "sumgenweight",
        "HEMCleaning",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "LepMiniIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ],
    "ele": [
        "sumgenweight",
        "HEMCleaning",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ],
    "lep": [
        "sumgenweight",
        "HEMCleaning",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "LepMiniIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ],
}


cut_to_label = {
    "sumgenweight": "sumgenweight",
    "HEMCleaning": "HEMCleaning",
    "Trigger": "Trigger",
    "METFilters": "METFilters",
    "OneLep": "n Leptons = 1",
    "NoTaus": "n Taus = 0",
    "LepIso": r"$\ell$ relative isolation",
    "LepMiniIso": r"$\ell$ mini-isolation",
    "OneCandidateJet": "n FatJets = 1",
    "CandidateJetpT": r"j $p_T > 250$GeV",
    "LepInJet": r"$\Delta R(j, \ell) < 0.8$",
    "JetLepOverlap": r"$\Delta R(j, \ell) > 0.03$",
    "dPhiJetMETCut": r"$\Delta \phi(\mathrm{MET}, j)<1.57$",
    # "$\mathrm{MET}>20~\GeV$"
}

parquet_to_latex = {
    "WJetsLNu": "$\PW(\Pell\PGn)$+",
    "QCD": "QCD",
    # "DYJets": "$\PZ(\Pell\Pell)$+jets",
    "TTbar": "\\ttbar",
    "Others": "Other MC",
    "ggF": "ggF",
    "VBF": "VBF",
    "VH": "VH",
    "ttH": "$t\\bar{t}H$",
    "Data": "Data",
}

# get lumi
with open("../fileset/luminosity.json") as f:
    luminosity = json.load(f)


def get_lumi(years, channels):
    lum_ = 0
    for year in years:
        lum = 0
        for ch in channels:
            lum += luminosity[ch][year] / 1000.0

        lum_ += lum / len(channels)
    return lum_


def get_sum_sumgenweight(pkl_files, year, sample):
    """Load and sum the sumgenweight of each pkl file."""

    sum_sumgenweight = 0
    for ifile in pkl_files:
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]

    return sum_sumgenweight


def get_xsecweight(pkl_files, year, ch, sample, is_data):
    """Get xsec-weight and scales events by lumi/sumgenweights."""

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
        xsec_weight = (xsec * luminosity[ch][year]) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


def get_cutflow(pkl_files, year, ch, sample, is_data):
    """Get cutflow from metadata but multiply by xsec-weight."""

    xsec_weight = get_xsecweight(pkl_files, year, ch, sample, is_data)

    if year == "2018":
        cuts = {
            "mu": ["sumgenweight", "HEMCleaning"],
            "ele": ["sumgenweight", "HEMCleaning"],
        }
    else:
        cuts = {
            "mu": ["sumgenweight"],
            "ele": ["sumgenweight"],
        }

    cuts["mu"] += [
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "LepMiniIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ]
    cuts["ele"] += [
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ]

    evyield = dict.fromkeys(cuts[ch], 0)
    for ik, pkl_file in enumerate(pkl_files):
        with open(pkl_file, "rb") as f:
            metadata = pkl.load(f)

        cutflows = metadata[sample][year]["cutflows"][ch]

        for key in evyield.keys():
            if key == "sumgenweight":
                evyield[key] += metadata[sample][year][key] * xsec_weight

            else:
                evyield[key] += cutflows[key] * xsec_weight
    return evyield


def make_cutflow_dict(years, channels, samples_dir, samples):
    cutflows = {}
    for year in years:
        print(f"Processing year {year}")

        cutflows[year] = {}

        for ch in channels:
            print(f"  {ch} channel")
            cutflows[year][ch] = {}

            condor_dir = os.listdir(samples_dir[year])

            for sample in condor_dir:
                # get a combined label to combine samples of the same process
                for key in utils.combine_samples:
                    if key in sample:
                        sample_to_use = utils.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample

                if sample_to_use not in samples:
                    continue

                is_data = False
                if sample_to_use == "Data":
                    is_data = True

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if sample_to_use not in cutflows[year][ch].keys():
                    cutflows[year][ch][sample_to_use] = get_cutflow(pkl_files, year, ch, sample, is_data)
                else:
                    temp = get_cutflow(pkl_files, year, ch, sample, is_data)
                    for key in cutflows[year][ch][sample_to_use]:
                        cutflows[year][ch][sample_to_use][key] += temp[key]

        print("------------------------------------------")
    return cutflows


def combine_channels(cutflows):
    """
    Must add lepminiso cutflow to electron channel.
    Will add to extra keys to the channels,
        1. `ele_new`: which contains the mini-isolation label to match the mu channel (the yield doesn't change)
        2. `lep`: which is the sum of `ele_new` and `mu`
    """

    common_cuts = [
        "sumgenweight",
        "HEMCleaning",
        "Trigger",
        "METFilters",
        "OneLep",
        "NoTaus",
        "LepIso",
        "LepMiniIso",
        "OneCandidateJet",
        "CandidateJetpT",
        "LepInJet",
        "JetLepOverlap",
        "dPhiJetMETCut",
    ]

    for year in cutflows.keys():
        cutflows[year]["ele_new"] = {}

        for sample in cutflows[year]["ele"].keys():
            cutflows[year]["ele_new"][sample] = {}

            for cut in common_cuts:
                if (year != "2018") and (cut == "HEMCleaning"):
                    continue

                if cut != "LepMiniIso":
                    cutflows[year]["ele_new"][sample][cut] = cutflows[year]["ele"][sample][cut]
                else:
                    cutflows[year]["ele_new"][sample][cut] = cutflows[year]["ele"][sample]["LepIso"]

    # combine both channels
    cutflows_new = {}
    for year in cutflows.keys():
        cutflows_new[year] = {}
        cutflows_new[year]["lep"] = {}

        for ch in ["mu", "ele_new"]:
            for sample in cutflows[year][ch]:
                if sample not in cutflows_new[year]["lep"]:
                    cutflows_new[year]["lep"][sample] = {}

                for cut in cutflows[year][ch][sample]:
                    if (year != "2018") and (cut == "HEMCleaning"):
                        continue

                    if cut not in cutflows_new[year]["lep"][sample]:
                        cutflows_new[year]["lep"][sample][cut] = cutflows[year][ch][sample][cut]
                    else:
                        cutflows_new[year]["lep"][sample][cut] += cutflows[year][ch][sample][cut]
        cutflows[year] = {**cutflows[year], **cutflows_new[year]}

    return cutflows


def combine_years(cutflows):
    """Combines all years under a key `Run2`. Will remove the HEM cleaning cutflow from 2018 first."""

    whatever_year = list(cutflows.keys())[0]
    channels = cutflows[whatever_year].keys()

    # combine all years
    cutflows_new = {}
    cutflows_new["Run2"] = {}

    for ch in channels:
        cutflows_new["Run2"][ch] = {}

        for year in cutflows:
            for sample in cutflows[year][ch]:
                if sample not in cutflows_new["Run2"][ch]:
                    cutflows_new["Run2"][ch][sample] = {}

                for cut in cutflows[year][ch][sample]:
                    if "HEM" in cut:
                        continue
                    if cut not in cutflows_new["Run2"][ch][sample]:
                        cutflows_new["Run2"][ch][sample][cut] = cutflows[year][ch][sample][cut]
                    else:
                        cutflows_new["Run2"][ch][sample][cut] += cutflows[year][ch][sample][cut]

    cutflows = {**cutflows, **cutflows_new}

    return cutflows


def combine_nondominant_bkg(cutflows):
    """Combines non-dominant backgrounds under key `Others`."""

    dominant_bkgs = ["WJetsLNu", "QCD", "TTbar"]
    signals = ["ggF", "VH", "VBF", "ttH"]

    for year in cutflows:
        for ch in cutflows[year]:
            cutflows[year][ch]["Others"] = dict.fromkeys(cutflows[year][ch]["WJetsLNu"], 0)
            for sample in cutflows[year][ch]:
                if sample == "Data":
                    continue
                if sample not in signals + dominant_bkgs:
                    for cut in cutflows[year][ch][sample]:
                        cutflows[year][ch]["Others"][cut] += cutflows[year][ch][sample][cut]
    return cutflows


def make_latex_cutflow_table(cutflows, year, ch, add_data=False, add_sumgenweight=False):
    """Will use the cutflows dictionary to make the LateX table we have in the AN."""

    samples_bkg = ["WJetsLNu", "QCD", "TTbar", "Others"]
    samples_sig = ["ggF", "VBF", "VH", "ttH"]

    # BACKGROUND
    headers = [parquet_to_latex[s] for s in samples_bkg]

    textabular = f"l{'r'*len(headers)}"
    textabular += "|r"

    texheader = "\\textbf{Inclusive Selection}" + " & " + " & ".join(headers) + " & Total MC "
    if add_data:
        textabular += "|r"
        texheader += "& Data "
    texheader += "\\\\"
    texdata = "\\hline\n"

    data = dict()

    for cut in cuts[ch]:
        if (year != "2018") and (cut == "HEMCleaning"):
            continue

        if not add_sumgenweight and cut == "sumgenweight":
            continue

        data[cut] = []

        for sample in samples_bkg:
            data[cut].append(round(cutflows[year][ch][sample][cut]))

        totalmc = 0
        for sample in samples_bkg + samples_sig:
            totalmc += round(cutflows[year][ch][sample][cut])
        data[cut].append(totalmc)

        if add_data:
            data[cut].append(round(cutflows[year][ch]["Data"][cut]))

    for label in data:
        if label == "z":
            texdata += "\\hline\n"
        texdata += f"{cut_to_label[label]} & {' & '.join(map(str,data[label]))} \\\\\n"

    texdata += "\\hline\n"

    # SIGNAL
    headers2 = [parquet_to_latex[s] for s in samples_sig]
    texheader2 = " & " + " & ".join(headers2) + "\\\\"
    texdata2 = "\\hline\n"

    data = dict()
    for cut in cuts[ch]:
        if (year != "2018") and (cut == "HEMCleaning"):
            continue

        data[cut] = []

        for sample in samples_sig:
            data[cut].append(round(cutflows[year][ch][sample][cut]))

    for label in data:
        if label == "z":
            texdata += "\\hline\n"
        texdata2 += f"{cut_to_label[label]} & {' & '.join(map(str,data[label]))} \\\\\n"

    # make table
    print("\\begin{table}[!htp]")
    print("\\begin{center}")

    print("\\begin{tabular}{" + textabular + "}")
    print(texheader)
    print(texdata, end="")
    print(texheader2)
    print(texdata2, end="")
    print("\\end{tabular}")

    if ch == "lep":
        print(
            "\\caption{Event yield of "
            + year
            + " Monte Carlo samples normalized to "
            + str(round(get_lumi([year], [ch])))
            + "\\fbinv.}"
        )
    else:
        print(
            "\\caption{Event yield of "
            + ch
            + " channel "
            + year
            + " Monte Carlo samples normalized to "
            + str(round(get_lumi([year], [ch])))
            + "\\fbinv.}"
        )

    print("\\label{sel-tab-cutflow" + year + "}")
    print("\\end{center}")
    print("\\end{table}")


def main(args):
    samples = ["ggF", "VH", "VBF", "ttH", "QCD", "DYJets", "WJetsLNu", "WZQQ", "TTbar", "SingleTop", "Diboson", "Data"]

    samples_dir = {
        "2016": "../eos/Dec7_2016",
        "2016APV": "../eos/Dec7_2016APV",
        "2017": "../eos/Dec7_2017",
        "2018": "../eos/Dec7_2018",
    }

    years = args.years.split(",")
    channels = args.channels.split(",")

    cutflows = make_cutflow_dict(years, channels, samples_dir, samples)

    if len(channels) > 1:
        cutflows = combine_channels(cutflows)

    cutflows = combine_years(cutflows)

    cutflows = combine_nondominant_bkg(cutflows)

    if len(channels) > 1:
        channel = "lep"
    else:
        channel = channels[0]

    if len(years) > 1:
        year = "Run2"
    else:
        year = years[0]

    make_latex_cutflow_table(cutflows, year, channel, add_data=args.add_data, add_sumgenweight=args.add_sumgenweight)


if __name__ == "__main__":
    # e.g.
    # python make_cutflow_table.py --years 2018,2017,2016,2016APV --channels ele,mu --add_data --add_sumgenweight

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--add_data", dest="add_data", help="adds an extra column for Data", action="store_true")
    parser.add_argument(
        "--add_sumgenweight", dest="add_sumgenweight", help="adds an extra row for the sumgenweight", action="store_true"
    )

    args = parser.parse_args()

    main(args)
