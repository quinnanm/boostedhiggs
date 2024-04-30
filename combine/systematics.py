from __future__ import division, print_function

import json
import logging
import warnings
from typing import List

import pandas as pd
import rhalphalib as rl
from utils import bkgs, samples, sigs

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

    # mini-isolation
    systs_dict["all_samples"]["miniisolation_SF"] = rl.NuisanceParameter("miniisolation_SF", "lnN")
    systs_dict_values["all_samples"]["miniisolation_SF"] = (1.02, 0.98)

    # tagger eff
    n = rl.NuisanceParameter("taggereff", "lnN")
    for sample in sigs:
        systs_dict[sample]["taggereff"] = n
        systs_dict_values[sample]["taggereff"] = (1.27, None)

    # QCD MC rate
    n = rl.NuisanceParameter("QCD_rate", "lnN")
    systs_dict["QCD"]["QCD_rate"] = n
    systs_dict_values["QCD"]["QCD_rate"] = (2, None)

    ############################################
    ############################################
    ############################################
    # Theory Systematics. (convention: https://gitlab.cern.ch/hh/naming-conventions#theory-uncertainties)

    # PDF
    n = rl.NuisanceParameter("pdf_Higgs_gg", "lnN")
    systs_dict["ggF"]["pdf_Higgs_gg"] = n
    systs_dict_values["ggF"]["pdf_Higgs_gg"] = (1.019, None)

    n = rl.NuisanceParameter("pdf_Higgs_qqHH", "lnN")
    systs_dict["VBF"]["pdf_Higgs_qqHH"] = n
    systs_dict_values["VBF"]["pdf_Higgs_qqHH"] = (1.021, 0.979)

    n = rl.NuisanceParameter("pdf_Higgs_ttH", "lnN")
    systs_dict["ttH"]["pdf_Higgs_ttH"] = n
    systs_dict_values["ttH"]["pdf_Higgs_ttH"] = (1.03, None)

    n = rl.NuisanceParameter("pdf_Higgs_qqbar", "lnN")
    systs_dict["WH"]["pdf_Higgs_qqbar"] = n
    systs_dict_values["WH"]["pdf_Higgs_qqbar"] = (1.017, None)
    systs_dict["ZH"]["pdf_Higgs_qqbar"] = n
    systs_dict_values["ZH"]["pdf_Higgs_qqbar"] = (1.013, None)

    # QCD scale
    n = rl.NuisanceParameter("QCDscale_ggH", "lnN")
    systs_dict["ggF"]["QCDscale_ggH"] = n
    systs_dict_values["ggF"]["QCDscale_ggH"] = (1.039, None)

    n = rl.NuisanceParameter("QCDscale_qqH", "lnN")
    systs_dict["VBF"]["QCDscale_qqH"] = n
    systs_dict_values["VBF"]["QCDscale_qqH"] = (1.004, 0.997)

    n = rl.NuisanceParameter("QCDscale_ttH", "lnN")
    systs_dict["ttH"]["QCDscale_ttH"] = n
    systs_dict_values["ttH"]["QCDscale_ttH"] = (1.058, 0.908)

    n = rl.NuisanceParameter("QCDscale_VH", "lnN")
    systs_dict["WH"]["QCDscale_VH"] = n
    systs_dict_values["WH"]["QCDscale_VH"] = (1.005, 0.993)
    systs_dict["ZH"]["QCDscale_VH"] = n
    systs_dict_values["ZH"]["QCDscale_VH"] = (1.038, 0.97)

    # alphas
    n = rl.NuisanceParameter("alpha_s", "lnN")
    systs_dict["ggF"]["alpha_s"] = n
    systs_dict_values["ggF"]["alpha_s"] = (1.026, None)
    systs_dict["VBF"]["alpha_s"] = n
    systs_dict_values["VBF"]["alpha_s"] = (1.005, None)
    systs_dict["ttH"]["alpha_s"] = n
    systs_dict_values["ttH"]["alpha_s"] = (1.020, None)
    systs_dict["WH"]["alpha_s"] = n
    systs_dict_values["WH"]["alpha_s"] = (1.009, None)
    systs_dict["ZH"]["alpha_s"] = n
    systs_dict_values["ZH"]["alpha_s"] = (1.009, None)

    return systs_dict, systs_dict_values


def systs_from_parquets(years):
    """
    Specify systematics that ARE stored in the parquets.

    The convention is:
        Dict[tuple[]]

        key: name of nuissance in the hist
        tuple[0]: name of the nuissance in the datacard
        tuple[1]: list of samples for ewhich the nuissance is applied

    """
    if len(years) == 4:
        yearlabel = "Run2"
    else:
        yearlabel = years[0]

    systs_from_parquets = {
        # systematics applied on all samples
        "weight_pileup": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_pileup_{yearlabel}", "shape"), sigs + bkgs),
        "weight_pileupIDSF": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PUIDSF_{yearlabel}", "shape"), sigs + bkgs),
        "weight_isolation_mu": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_isolation_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        "weight_isolation_ele": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_isolation_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        "weight_id_mu": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_identification_{yearlabel}", "lnN"), sigs + bkgs),
        "weight_id_ele": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_identification_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        "weight_reco_ele": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_reconstruction_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        "weight_L1Prefiring": (rl.NuisanceParameter(f"C{CMS_PARAMS_LABEL}L1Prefiring_{yearlabel}", "lnN"), sigs + bkgs),
        "weight_trigger_ele": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_trigger_{yearlabel}", "lnN"), sigs + bkgs),
        "weight_trigger_iso_mu": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_iso_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        "weight_trigger_noniso_mu": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_{yearlabel}", "lnN"),
            sigs + bkgs,
        ),
        # systematics applied only on ggF/VBF
        "weight_PSFSR": (rl.NuisanceParameter("PSFSR", "shape"), ["ggF", "VBF"]),
        "weight_PSISR": (rl.NuisanceParameter("PSISR", "shape"), ["ggF", "VBF"]),
        "rec_higgs_mUES": (rl.NuisanceParameter("AK8_UES", "shape"), ["ggF", "VBF"]),
        "rec_higgs_mJES": (rl.NuisanceParameter("AK8_JES", "shape"), ["ggF", "VBF"]),
        "rec_higgs_mJER": (rl.NuisanceParameter("AK8_JER", "shape"), ["ggF", "VBF"]),
        "rec_higgs_mJMS": (rl.NuisanceParameter("AK8_JMS", "shape"), ["ggF", "VBF"]),
        "rec_higgs_mJMR": (rl.NuisanceParameter("AK8_JMR", "shape"), ["ggF", "VBF"]),
        # systematics applied only on WJets
        "weight_d1K_NLO": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{yearlabel}", "lnN"), ["WJetsLNu"]),
        "weight_d2K_NLO": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{yearlabel}", "lnN"), ["WJetsLNu"]),
        "weight_d3K_NLO": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{yearlabel}", "lnN"), ["WJetsLNu"]),
        "weight_d1kappa_EW": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{yearlabel}", "lnN"), ["WJetsLNu"]),
        "weight_W_d2kappa_EW": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{yearlabel}", "lnN"),
            ["WJetsLNu"],
        ),
        "weight_W_d3kappa_EW": (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{yearlabel}", "lnN"),
            ["WJetsLNu"],
        ),
        # acceptance unc.
        "weight_pdf_acceptance": (rl.NuisanceParameter("PDFscale_acc", "shape"), ["ggF", "VBF", "WH", "ZH"]),
        "weight_qcd_scale": (
            rl.NuisanceParameter("QCDscale_acc", "shape"),
            ["ggF", "VBF", "WH", "ZH", "WJetsLNu", "TTbar"],
        ),
        # "weight_d1kappa_EW": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{yearlabel}", "lnN"), ["DYJets"]), # noqa
        # "weight_Z_d2kappa_EW": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{yearlabel}", "lnN"), ["DYJets"]), # noqa
        # "weight_Z_d3kappa_EW": (rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{yearlabel}", "lnN"), ["DYJets"]), # noqa
    }

    # add btag SF
    for year in years:
        systs_from_parquets[f"weight_btagSFlight{year}"] = (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFlight_{year}", "lnN"),
            sigs + bkgs,
        )

        systs_from_parquets[f"weight_btagSFbc{year}"] = (
            rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbc_{year}", "lnN"),
            sigs + bkgs,
        )

    systs_from_parquets["weight_btagSFlightCorrelated"] = (
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFlightCorrelated", "lnN"),
        sigs + bkgs,
    )
    systs_from_parquets[sigs + bkgs]["weight_btagSFbcCorrelated"] = (
        rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbcCorrelated", "lnN"),
        sigs + bkgs,
    )

    return systs_from_parquets
