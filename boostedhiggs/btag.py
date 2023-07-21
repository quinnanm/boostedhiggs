import importlib.resources
import pathlib

import awkward as ak
import correctionlib
import numpy as np
from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import JetArray

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
}


def get_UL_year(year):
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"
    return f"{year}_UL"


def get_pog_json(obj, year):
    try:
        pog_json = pog_jsons[obj]
    except ValueError:
        print(f"No json for {obj}")
    year = get_UL_year(year)
    return f"{pog_correction_path}POG/{pog_json[0]}/{year}/{pog_json[1]}"
    # noqa: os.system(f"cp {pog_correction_path}POG/{pog_json[0]}/{year}/{pog_json[1]} boostedhiggs/data/POG_{pog_json[0]}_{year}_{pog_json[1]}")
    # fname = ""
    # with importlib.resources.path("boostedhiggs.data", f"POG_{pog_json[0]}_{year}_{pog_json[1]}") as filename:
    #     fname = str(filename)
    # print(fname)
    # return fname


"""
Lepton Scale Factors
----

Muons:
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018

- UL CorrectionLib html files:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/MUO_muon_Z_Run2_UL/
  e.g. one example of the correction json files can be found here:
  https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/raw/master/Run2/UL/2017/2017_trigger/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers_schemaV2.json
  - Trigger iso and non-iso
  - Isolation: We use RelIso<0.25 (LooseRelIso) with medium prompt ID
  - Reconstruction ID: We use medium prompt ID

Electrons:
- UL CorrectionLib htmlfiles:
  https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/EGM_electron_Run2_UL/
  - ID: wp90noiso
  - Reconstruction: RecoAbove20
  - Trigger:
    Looks like the EGM group does not provide them but applying these for now.
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements (derived by Siqi Yuan)
    These include 2017: (HLT_ELE35 OR HLT_ELE115 OR HLT_Photon200)
    and 2018: (HLT_Ele32 OR HLT_ELE115 OR HLT_Photon200)
  - Isolation:
    No SFs for RelIso?
"""

lepton_corrections = {
    "trigger_iso": {
        "muon": {  # For IsoMu24 (| IsoTkMu24 )
            "2016APV": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # preVBP
            "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # postVBF
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        },
    },
    "trigger_noniso": {
        "muon": {  # For Mu50 (| TkMu50 )
            "2016APV": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2016": "NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2017": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
            "2018": "NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose",
        },
    },
    "isolation": {
        "muon": {
            "2016APV": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2016": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2017": "NUM_LooseRelIso_DEN_MediumPromptID",
            "2018": "NUM_LooseRelIso_DEN_MediumPromptID",
        },
        # "electron": {
        # },
    },
    # NOTE: We do not have SFs for mini-isolation yet
    "id": {
        "muon": {
            "2016APV": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2016": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2017": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2018": "NUM_MediumPromptID_DEN_TrackerMuons",
        },
        # NOTE: should check that we do not have electrons w pT>500 GeV (I do not think we do)
        "electron": {
            "2016APV": "wp90noiso",
            "2016": "wp90noiso",
            "2017": "wp90noiso",
            "2018": "wp90noiso",
        },
    },
    "reco": {
        "electron": {
            "2016APV": "RecoAbove20",
            "2016": "RecoAbove20",
            "2017": "RecoAbove20",
            "2018": "RecoAbove20",
        },
    },
}


def add_lepton_weight(weights, lepton, year, lepton_type="muon"):
    ul_year = get_UL_year(year)
    if lepton_type == "electron":
        ul_year = ul_year.replace("_UL", "")
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(lepton_type, year))

    def set_isothreshold(corr, value, lepton_pt, lepton_type):
        iso_threshold = {
            "muon": 55.0,
            "electron": 120.0,
        }[lepton_type]
        if corr == "trigger_iso":
            value[lepton_pt > iso_threshold] = 1.0
        elif corr == "trigger_noniso":
            value[lepton_pt < iso_threshold] = 1.0
        elif corr == "isolation":
            value[lepton_pt > iso_threshold] = 1.0
        return value

    def get_clip(lep_pt, lep_eta, lepton_type, corr=None):
        clip_pt = [0.0, 2000]
        clip_eta = [-2.4999, 2.4999]
        if lepton_type == "electron":
            clip_pt = [10.0, 499.999]
            if corr == "reco":
                clip_pt = [20.1, 499.999]
        elif lepton_type == "muon":
            clip_pt = [30.0, 1000.0]
            clip_eta = [0.0, 2.3999]
            if corr == "trigger_noniso":
                clip_pt = [52.0, 1000.0]
        lepton_pt = np.clip(lep_pt, clip_pt[0], clip_pt[1])
        lepton_eta = np.clip(lep_eta, clip_eta[0], clip_eta[1])
        return lepton_pt, lepton_eta

    lep_pt = np.array(ak.fill_none(lepton.pt, 0.0))
    lep_eta = np.array(ak.fill_none(lepton.eta, 0.0))
    if lepton_type == "muon":
        lep_eta = np.abs(lep_eta)

    for corr, corrDict in lepton_corrections.items():
        if lepton_type not in corrDict.keys():
            continue
        if year not in corrDict[lepton_type].keys():
            continue
        json_map_name = corrDict[lepton_type][year]

        lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)

        values = {}
        if lepton_type == "muon":
            values["nominal"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "sf")
        else:
            values["nominal"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sf", json_map_name, lepton_eta, lepton_pt)

        if lepton_type == "muon":
            values["up"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "systup")
            values["down"] = cset[json_map_name].evaluate(ul_year, lepton_eta, lepton_pt, "systdown")
        else:
            values["up"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfup", json_map_name, lepton_eta, lepton_pt)
            values["down"] = cset["UL-Electron-ID-SF"].evaluate(ul_year, "sfdown", json_map_name, lepton_eta, lepton_pt)

        for key, val in values.items():
            # restrict values to 1 for some SFs if we are above/below the ISO threshold
            values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

        # add weights (for now only the nominal weight)
        weights.add(f"{corr}_{lepton_type}", values["nominal"], values["up"], values["down"])

    # quick hack to add electron trigger SFs
    if lepton_type == "electron":
        corr = "trigger"
        with importlib.resources.path("boostedhiggs.data", f"electron_trigger_{ul_year}_UL.json") as filename:
            cset = correctionlib.CorrectionSet.from_file(str(filename))
            lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)
            # stil need to add uncertanties..
            values["nominal"] = cset["UL-Electron-Trigger-SF"].evaluate(lepton_eta, lepton_pt)
            # print(values["nominal"][lep_pt>30])
            weights.add(f"{corr}_{lepton_type}", values["nominal"])


def add_pileup_weight(weights, year, mod, nPU):
    """
    Should be able to do something similar to lepton weight but w pileup
    e.g. see here: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
    """
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year + mod))

    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }

    values = {}

    values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
    values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

    # add weights (for now only the nominal weight)
    weights.add("pileup", values["nominal"], values["up"], values["down"])


def _btagSF(cset, jets, flavour, wp="M", algo="deepJet", syst="central"):
    j, nj = ak.flatten(jets), ak.num(jets)
    corrs = cset[f"{algo}_comb"] if flavour == "bc" else cset[f"{algo}_incl"]
    sf = corrs.evaluate(
        syst,
        wp,
        np.array(j.hadronFlavour),
        np.array(abs(j.eta)),
        np.array(j.pt),
    )
    return ak.unflatten(sf, nj)


def _btag_prod(eff, sf):
    num = ak.fill_none(ak.prod(1 - sf * eff, axis=-1), 1)
    den = ak.fill_none(ak.prod(1 - eff, axis=-1), 1)
    return num, den


def add_btag_weights(
    weights: Weights,
    year: str,
    jets: JetArray,
    jet_selector: ak.Array,
    wp: str = "M",
    algo: str = "deepJet",
):
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))
    efflookup = cutil.load(package_path + f"/corrections/btag_effs/btageff_deepJet_M_{year}.coffea")

    lightJets = jets[jet_selector & (jets.hadronFlavour == 0)]
    bcJets = jets[jet_selector & (jets.hadronFlavour > 0)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    lightnum, lightden = _btag_prod(lightEff, lightSF)
    bcnum, bcden = _btag_prod(bcEff, bcSF)

    weight = np.nan_to_num((1 - lightnum * bcnum) / (1 - lightden * bcden), nan=1)
    weights.add("btagSF", weight)
