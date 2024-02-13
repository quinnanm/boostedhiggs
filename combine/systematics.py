from __future__ import division, print_function

import json
import logging
import warnings
from typing import List

import pandas as pd
import rhalphalib as rl
from utils import samples

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def systs_not_from_parquets(years: List[str], lep_channels: List[str]):
    """
    Define systematics that are NOT stored in the parquets.

    Args
        years: e.g. ["2018", "2017", "2016APV", "2016"]
        lep_channels: e.g. ["mu", "ele"]

    Returns
        systs_dict [dict]:        keys are systematics; values are rl.NuisanceParameters
        systs_dict_values [dict]: keys are same as above; values are tuples (up, down) and if (up, None) then down=up
    """
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in lep_channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(lep_channels)

    # get the LUMI covered in the templates
    full_lumi = 0
    for year in years:
        full_lumi += LUMI[year]

    systs_dict, systs_dict_values = {}, {}

    # COMMON SYSTEMATICS
    systs_dict["all_samples"], systs_dict_values["all_samples"] = {}, {}

    # branching ratio systematics
    systs_dict["all_samples"]["BR_hww"] = rl.NuisanceParameter("BR_hww", "lnN")
    systs_dict_values["all_samples"]["BR_hww"] = (1.0153, 0.9848)

    # lumi systematics
    if "2016" in years:
        systs_dict["all_samples"]["lumi_13TeV_2016"] = rl.NuisanceParameter("CMS_lumi_13TeV_2016", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2016"] = (1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi), None)
    if "2017" in years:
        systs_dict["all_samples"]["lumi_13TeV_2017"] = rl.NuisanceParameter("CMS_lumi_13TeV_2017", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2017"] = (1.02 ** (LUMI["2017"] / full_lumi), None)

    if "2018" in years:
        systs_dict["all_samples"]["lumi_13TeV_2018"] = rl.NuisanceParameter("CMS_lumi_13TeV_2018", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2018"] = (1.015 ** (LUMI["2018"] / full_lumi), None)

    if len(years) == 4:
        systs_dict["all_samples"]["lumi_13TeV_correlated"] = rl.NuisanceParameter("CMS_lumi_13TeV_corelated", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_correlated"] = (
            (
                (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
                * (1.009 ** (LUMI["2017"] / full_lumi))
                * (1.02 ** (LUMI["2018"] / full_lumi))
            ),
            None,
        )

        systs_dict["all_samples"]["lumi_13TeV_1718"] = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_1718"] = (
            (1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi)),
            None,
        )

    # PER SAMPLE SYSTEMATICS
    for sample in samples:
        systs_dict[sample], systs_dict_values[sample] = {}, {}

    # tagger eff
    n = rl.NuisanceParameter("taggereff", "lnN")
    for sample in ["ggF", "VBF", "ttH", "WH", "ZH"]:
        systs_dict[sample]["taggereff"] = n
        systs_dict_values[sample]["taggereff"] = (1.30, None)

    # QCD scale
    n = rl.NuisanceParameter("QCD_scale", "lnN")
    systs_dict["ggF"]["QCD_scale"] = n
    systs_dict_values["ggF"]["QCD_scale"] = (1.039, 0.961)
    systs_dict["VBF"]["QCD_scale"] = n
    systs_dict_values["VBF"]["QCD_scale"] = (1.004, 0.997)
    systs_dict["ttH"]["QCD_scale"] = n
    systs_dict_values["ttH"]["QCD_scale"] = (1.058, 0.908)

    systs_dict["WH"]["QCD_scale"] = n
    systs_dict_values["WH"]["QCD_scale"] = (1.005, 0.993)
    systs_dict["ZH"]["QCD_scale"] = n
    systs_dict_values["ZH"]["QCD_scale"] = (1.038, 0.97)

    # PDF scale
    n = rl.NuisanceParameter("PDF_scale", "lnN")
    systs_dict["ggF"]["PDF_scale"] = n
    systs_dict_values["ggF"]["PDF_scale"] = (1.019, 0.981)
    systs_dict["VBF"]["PDF_scale"] = n
    systs_dict_values["VBF"]["PDF_scale"] = (1.021, 0.979)
    systs_dict["ttH"]["PDF_scale"] = n
    systs_dict_values["ttH"]["PDF_scale"] = (1.03, 0.97)

    systs_dict["WH"]["PDF_scale"] = n
    systs_dict_values["WH"]["PDF_scale"] = (1.017, 0.983)
    systs_dict["ZH"]["PDF_scale"] = n
    systs_dict_values["ZH"]["PDF_scale"] = (1.013, 0.987)

    return systs_dict, systs_dict_values


def systs_from_parquets(years):
    """
    Specify systematics that ARE stored in the parquets
    """
    if len(years) == 4:
        year = "Run2"
    else:
        year = years[0]

    systs_from_parquets = {
        "all_samples": {
            # "weight_mu_btagSFlight_2017": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFlight_{year}", "lnN"
            #     ),
            # "weight_mu_btagSFlight_correlated": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFlight_correlated", "lnN"
            # ),
            # "weight_mu_btagSFbc_2017": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbc_{year}", "lnN"),
            # "weight_mu_btagSFbc_correlated": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFbc_correlated", "lnN"
            # ),
            # "weight_pileup": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PU_{year}", "shape"),
            # "weight_pileupIDSF": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PUIDSF_{year}", "shape"),
            "weight_isolation": rl.NuisanceParameter(f"CMS_iso_{year}", "lnN"),
            "weight_id": rl.NuisanceParameter(f"CMS_id_{year}", "lnN"),
            "weight_reco_ele": rl.NuisanceParameter("CMS_reconstruction_ele", "lnN"),
            "weight_L1Prefiring": rl.NuisanceParameter(f"CMS_L1Prefiring_{year}", "lnN"),
            "weight_trigger_ele": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_trigger_{year}_ele", "lnN"),
            "weight_trigger_iso_mu": rl.NuisanceParameter(f"CMS_mu_trigger_iso_{year}_mu", "lnN"),
            "weight_trigger_noniso_mu": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_{year}_mu", "lnN"),
            # shape weights
            "fj_ptJES": rl.NuisanceParameter("AK8_Jet_pt_JES", "shape"),
            "fj_ptJER": rl.NuisanceParameter("AK8_Jet_pt_JER", "shape"),
            "fj_massJMS": rl.NuisanceParameter("AK8_Jet_mass_JMS", "shape"),
            "fj_massJMR": rl.NuisanceParameter("AK8_Jet_mass_JMR", "shape"),
            "mjjJES": rl.NuisanceParameter("mjj_JES", "shape"),
            "mjjJER": rl.NuisanceParameter("mjj_JER", "shape"),
            "rec_higgs_mUES": rl.NuisanceParameter("Higgs_candidate_mass_UES", "shape"),
            "rec_higgs_mJES": rl.NuisanceParameter("Higgs_candidate_mass_JES", "shape"),
            "rec_higgs_mJER": rl.NuisanceParameter("Higgs_candidate_mass_JER", "shape"),
            "rec_higgs_mJMS": rl.NuisanceParameter("Higgs_candidate_mass_JMS", "shape"),
            "rec_higgs_mJMR": rl.NuisanceParameter("Higgs_candidate_mass_JMR", "shape"),
            "rec_higgs_ptUES": rl.NuisanceParameter("Higgs_candidate_pt_UES", "shape"),
            "rec_higgs_ptJES": rl.NuisanceParameter("Higgs_candidate_pt_JES", "shape"),
            "rec_higgs_ptJER": rl.NuisanceParameter("Higgs_candidate_pt_JER", "shape"),
            "rec_higgs_ptJMS": rl.NuisanceParameter("Higgs_candidate_pt_JMS", "shape"),
            "rec_higgs_ptJMR": rl.NuisanceParameter("Higgs_candidate_pt_JMR", "shape"),
        },
        # signal
        "ggF": {
            "weight_PSFSR": rl.NuisanceParameter("PSFSR_ggF", "shape"),
            "weight_PSISR": rl.NuisanceParameter("PSISR_ggF", "shape"),
        },
        "VBF": {
            "weight_PSFSR": rl.NuisanceParameter("PSFSR_VBF", "shape"),
            "weight_PSISR": rl.NuisanceParameter("PSISR_VBF", "shape"),
        },
        "ttH": {},
        "WH": {},
        "ZH": {},
        # bkgs
        "QCD": {},
        "DYJets": {
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{year}", "lnN"),
            "weight_Z_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{year}", "lnN"),
            "weight_Z_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{year}", "lnN"),
        },
        "WJetsLNu": {
            "weight_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{year}", "lnN"),
            "weight_d2K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{year}", "lnN"),
            "weight_d3K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{year}", "lnN"),
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{year}", "lnN"),
            "weight_W_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{year}", "lnN"),
            "weight_W_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{year}", "lnN"),
        },
        "TTbar": {},
        "SingleTop": {},
        "Diboson": {},
        "WZQQ": {},
        "EWKvjets": {},
    }

    return systs_from_parquets
