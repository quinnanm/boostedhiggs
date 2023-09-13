import importlib.resources
import pickle
import warnings
from typing import Dict

import awkward as ak
import correctionlib
import numpy as np
from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import JetArray

btagWPs = {
    "deepJet": {
        "2016APV": {
            "L": 0.0508,
            "M": 0.2598,
            "T": 0.6502,
        },
        "2016": {
            "L": 0.0480,
            "M": 0.2489,
            "T": 0.6377,
        },
        "2017": {
            "L": 0.0532,
            "M": 0.3040,
            "T": 0.7476,
        },
        "2018": {
            "L": 0.0490,
            "M": 0.2783,
            "T": 0.7100,
        },
    },
    "deepCSV": {
        "2016APV": {
            "L": 0.2027,
            "M": 0.6001,
            "T": 0.8819,
        },
        "2016": {
            "L": 0.1918,
            "M": 0.5847,
            "T": 0.8767,
        },
        "2017": {
            "L": 0.1355,
            "M": 0.4506,
            "T": 0.7738,
        },
        "2018": {
            "L": 0.1208,
            "M": 0.4168,
            "T": 0.7665,
        },
    },
}

with importlib.resources.path("boostedhiggs.data", "msdcorr.json") as filename:
    msdcorr = correctionlib.CorrectionSet.from_file(str(filename))


def corrected_msoftdrop(fatjets):
    msdraw = np.sqrt(
        np.maximum(
            0.0,
            (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum().mass2,
        )
    )
    # msoftdrop = fatjets.msoftdrop
    msdfjcorr = msdraw / (1 - fatjets.rawFactor)

    corr = msdcorr["msdfjcorr"].evaluate(
        np.array(ak.flatten(msdfjcorr / fatjets.pt)),
        np.array(ak.flatten(np.log(fatjets.pt))),
        np.array(ak.flatten(fatjets.eta)),
    )
    corr = ak.unflatten(corr, ak.num(fatjets))
    corrected_mass = msdfjcorr * corr

    return corrected_mass


with importlib.resources.path("boostedhiggs.data", "ULvjets_corrections.json") as filename:
    vjets_kfactors = correctionlib.CorrectionSet.from_file(str(filename))


def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24)) & genpart.hasFlags(["fromHardProcess", "isLastCopy"])]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))


def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    znlosysts = [
        "d1kappa_EW",
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(
                syst,
                ones,
                ewkcorr.evaluate(syst + "_up", vpt) / ewknom,
                ewkcorr.evaluate(syst + "_down", vpt) / ewknom,
            )

    if "ZJetsToQQ_HT" in dataset or "DYJetsToLL_M-" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)

    elif "DYJetsToLL_Pt" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = None
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(znlosysts, qcdcorr, ewkcorr, vpt)

    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu" in dataset:
        vpt = get_vpt(genpart)
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)


def add_pdf_weight(weights, pdf_weights):
    nweights = len(weights.weight())
    nom = np.ones(nweights)
    up = np.ones(nweights)
    down = np.ones(nweights)
    # docstring = pdf_weights.__doc__

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
    if True:
        # Hessian PDF weights
        # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        arg = pdf_weights[:, 1:-2] - np.ones((nweights, 100))
        summed = ak.sum(np.square(arg), axis=1)
        pdf_unc = np.sqrt((1.0 / 99.0) * summed)
        weights.add("PDF_weight", nom, pdf_unc + nom)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5 * (pdf_weights[:, 102] - pdf_weights[:, 101])
        weights.add("aS_weight", nom, as_unc + nom)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        weights.add("PDFaS_weight", nom, pdfas_unc + nom)

    else:
        weights.add("aS_weight", nom, up, down)
        weights.add("PDF_weight", nom, up, down)
        weights.add("PDFaS_weight", nom, up, down)


# 7-point scale variations
def add_scalevar_7pt(weights, var_weights):
    # docstring = var_weights.__doc__
    nweights = len(weights.weight())

    nom = np.ones(nweights)
    up = np.ones(nweights)
    down = np.ones(nweights)

    if len(var_weights) > 0:
        if len(var_weights[0]) == 9:
            # you skip the extremes, where one (uR, uF) is multiplied by 2 and the other by 0.5
            up = np.maximum.reduce(
                [
                    var_weights[:, 0],
                    var_weights[:, 1],
                    var_weights[:, 3],
                    var_weights[:, 5],
                    var_weights[:, 7],
                    var_weights[:, 8],
                ]
            )
            down = np.minimum.reduce(
                [
                    var_weights[:, 0],
                    var_weights[:, 1],
                    var_weights[:, 3],
                    var_weights[:, 5],
                    var_weights[:, 7],
                    var_weights[:, 8],
                ]
            )
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))
    weights.add("scalevar_7pt", nom, up, down)


# 3-point scale variations
def add_scalevar_3pt(weights, var_weights):
    # docstring = var_weights.__doc__

    nweights = len(weights.weight())

    nom = np.ones(nweights)
    up = np.ones(nweights)
    down = np.ones(nweights)

    if len(var_weights) > 0:
        if len(var_weights[0]) == 9:
            up = np.maximum(var_weights[:, 0], var_weights[:, 8])
            down = np.minimum(var_weights[:, 0], var_weights[:, 8])
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))

    weights.add("scalevar_3pt", nom, up, down)


def add_ps_weight(weights, ps_weights):
    nweights = len(weights.weight())
    nom = np.ones(nweights)
    up_isr = np.ones(nweights)
    down_isr = np.ones(nweights)
    up_fsr = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if ps_weights is not None:
        if len(ps_weights[0]) == 4:
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
        else:
            warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")
    weights.add("UEPS_ISR", nom, up_isr, down_isr)
    weights.add("UEPS_FSR", nom, up_fsr, down_fsr)


with importlib.resources.path("boostedhiggs.data", "EWHiggsCorrections.json") as filename:
    hew_kfactors = correctionlib.CorrectionSet.from_file(str(filename))


def add_HiggsEW_kFactors(weights, genpart, dataset):
    """EW Higgs corrections"""

    def get_hpt():
        boson = ak.firsts(genpart[(genpart.pdgId == 25) & genpart.hasFlags(["fromHardProcess", "isLastCopy"])])
        return np.array(ak.fill_none(boson.pt, 0.0))

    if "VBF" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VBF_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VBF_EW", ewknom)

    if "WplusH" in dataset or "WminusH" in dataset or "ZH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VH_EW", ewknom)

    if "ttH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["ttH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("ttH_EW", ewknom)


def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask

    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)


lumi_masks = {
    "2016": build_lumimask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
    "2017": build_lumimask("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
    "2018": build_lumimask("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
}


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "jec": ["JME", "fatJet_jerc.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
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


def add_btag_weights_farouk(year: str, jets: JetArray, jet_selector: ak.Array, wp: str = "M", algo: str = "deepJet"):
    """
    Following https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1b_Event_reweighting_using_scale

    """

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

    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))

    ul_year = get_UL_year(year)
    with importlib.resources.path("boostedhiggs.data", f"btageff_{algo}_{wp}_{ul_year}.coffea") as filename:
        efflookup = cutil.load(filename)

    lightJets = jets[jet_selector & (jets.hadronFlavour == 0) & (abs(jets.eta) < 2.5)]
    bcJets = jets[jet_selector & (jets.hadronFlavour > 0) & (abs(jets.eta) < 2.5)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    light_probs = ak.fill_none(ak.prod(1 - lightSF * lightEff, axis=-1), 1)
    bc_probs = ak.fill_none(ak.prod(1 - bcSF * bcEff, axis=-1), 1)

    weight = light_probs * bc_probs

    # weight(>0 btag) = 1 - weight(0 btag)
    weight = 1 - weight

    return weight


def add_btag_weights(
    weights: Weights,
    year: str,
    jets: JetArray,
    jet_selector: ak.Array,
    wp: str = "M",
    algo: str = "deepJet",
):
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

    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))

    ul_year = get_UL_year(year)
    with importlib.resources.path("boostedhiggs.data", f"btageff_{algo}_{wp}_{ul_year}.coffea") as filename:
        efflookup = cutil.load(filename)

    lightJets = jets[jet_selector & (jets.hadronFlavour == 0) & (abs(jets.eta) < 2.5)]
    bcJets = jets[jet_selector & (jets.hadronFlavour > 0) & (abs(jets.eta) < 2.5)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    lightSFUp = _btagSF(cset, lightJets, "light", wp, algo, syst="up")
    lightSFDown = _btagSF(cset, lightJets, "light", wp, algo, syst="down")
    lightSFUpCorr = _btagSF(cset, lightJets, "light", wp, algo, syst="up_correlated")
    lightSFDownCorr = _btagSF(cset, lightJets, "light", wp, algo, syst="down_correlated")
    bcSFUp = _btagSF(cset, bcJets, "bc", wp, algo, syst="up")
    bcSFDown = _btagSF(cset, bcJets, "bc", wp, algo, syst="down")
    bcSFUpCorr = _btagSF(cset, bcJets, "bc", wp, algo, syst="up_correlated")
    bcSFDownCorr = _btagSF(cset, bcJets, "bc", wp, algo, syst="down_correlated")

    def _get_weight(lightEff, lightSF, bcEff, bcSF):
        lightnum, lightden = _btag_prod(lightEff, lightSF)
        bcnum, bcden = _btag_prod(bcEff, bcSF)
        weight = np.nan_to_num((1 - lightnum * bcnum) / (1 - lightden * bcden), nan=1)
        return weight

    weight = _get_weight(lightEff, lightSF, bcEff, bcSF)
    weights.add("btagSF", weight)

    # add systematics
    nominal = np.ones(len(weight))
    weights.add(
        f"btagSFlight_{year}",
        nominal,
        weightUp=_get_weight(lightEff, lightSFUp, bcEff, bcSF),
        weightDown=_get_weight(lightEff, lightSFDown, bcEff, bcSF),
    )
    weights.add(
        f"btagSFbc_{year}",
        nominal,
        weightUp=_get_weight(lightEff, lightSF, bcEff, bcSFUp),
        weightDown=_get_weight(lightEff, lightSF, bcEff, bcSFDown),
    )
    weights.add(
        "btagSFlight_correlated",
        nominal,
        weightUp=_get_weight(lightEff, lightSFUpCorr, bcEff, bcSF),
        weightDown=_get_weight(lightEff, lightSFDownCorr, bcEff, bcSF),
    )
    weights.add(
        "btagSFbc_correlated",
        nominal,
        weightUp=_get_weight(lightEff, lightSF, bcEff, bcSFUpCorr),
        weightDown=_get_weight(lightEff, lightSF, bcEff, bcSFDownCorr),
    )


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
  - ID and Isolation:
    - wp90noiso for high pT electrons
    - wp90iso for low pT electrons
  - Reconstruction: RecoAbove20
  - Trigger: Derived using EGamma recommendation: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements
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
        "electron": {
            "2016APV": "wp90iso",
            "2016": "wp90iso",
            "2017": "wp90iso",
            "2018": "wp90iso",
        },
    },
    "id": {
        "muon": {
            "2016APV": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2016": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2017": "NUM_MediumPromptID_DEN_TrackerMuons",
            "2018": "NUM_MediumPromptID_DEN_TrackerMuons",
        },
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
        """
        restrict values to 1 for some SFs if we are above/below the ISO threshold
        """
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
        elif corr == "id" and lepton_type == "electron":
            value[lepton_pt < iso_threshold] = 1.0
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
            values[key] = set_isothreshold(corr, val, np.array(ak.fill_none(lepton.pt, 0.0)), lepton_type)

        # add weights (for now only the nominal weight)
        weights.add(f"{corr}_{lepton_type}", values["nominal"], values["up"], values["down"])

    # quick hack to add electron trigger SFs
    if lepton_type == "electron":
        corr = "trigger"
        with importlib.resources.path("boostedhiggs.data", f"electron_trigger_{ul_year}_UL.json") as filename:
            cset = correctionlib.CorrectionSet.from_file(str(filename))
            lepton_pt, lepton_eta = get_clip(lep_pt, lep_eta, lepton_type, corr)
            values["nominal"] = cset["UL-Electron-Trigger-SF"].evaluate(
                ul_year + "_UL", "sf", "trigger", lepton_eta, lepton_pt
            )
            values["up"] = cset["UL-Electron-Trigger-SF"].evaluate(ul_year + "_UL", "sfup", "trigger", lepton_eta, lepton_pt)
            values["down"] = cset["UL-Electron-Trigger-SF"].evaluate(
                ul_year + "_UL", "sfdown", "trigger", lepton_eta, lepton_pt
            )
            weights.add(f"{corr}_{lepton_type}", values["nominal"], values["up"], values["down"])


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

    # add weights
    weights.add("pileup", values["nominal"], values["up"], values["down"])


# find corrections path using this file's path
try:
    with importlib.resources.path("boostedhiggs.data", "jec_compiled.pkl") as filename:
        with open(filename, "rb") as filehandler:
            jmestuff = pickle.load(filehandler)
        ak4jet_factory = jmestuff["jet_factory"]
        fatjet_factory = jmestuff["fatjet_factory"]
        met_factory = jmestuff["met_factory"]
except KeyError:
    print("Failed loading compiled JECs")


def _add_jec_variables(jets: JetArray, event_rho: ak.Array) -> JetArray:
    """add variables needed for JECs"""
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    # gen pT needed for smearing
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def get_jec_jets(
    events,
    jets,
    year: str,
    isData: bool = False,
    jecs: Dict[str, str] = None,
    fatjets: bool = True,
):
    """
    Based on https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    Eventually update to V5 JECs once I figure out what's going on with the 2017 UL V5 JER scale factors

    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/

    If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
    """

    jec_vars = ["pt"]  # variables we are saving that are affected by JECs
    if fatjets:
        jet_factory = fatjet_factory
    else:
        jet_factory = ak4jet_factory

    apply_jecs = not (not ak.any(jets.pt) or isData)

    import cachetools

    jec_cache = cachetools.Cache(np.inf)

    corr_key = f"{get_UL_year(year)}mc".replace("_UL", "")

    # fatjet_factory.build gives an error if there are no fatjets in event
    if apply_jecs:
        jets = jet_factory[corr_key].build(_add_jec_variables(jets, events.fixedGridRhoFastjetAll), jec_cache)

    # return only fatjets if no jecs given
    if jecs is None:
        return jets

    jec_shifted_vars = {}

    for jec_var in jec_vars:
        tdict = {"": jets[jec_var]}
        if apply_jecs:
            for key, shift in jecs.items():
                for var in ["up", "down"]:
                    tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

        jec_shifted_vars[jec_var] = tdict

    return jets, jec_shifted_vars
