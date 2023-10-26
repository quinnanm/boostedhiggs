"""
Creates "combine datacards" using hist.Hist templates, and
sets up data-driven QCD background estimate ('rhalphabet' method)

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py

Author: Farouk Mokhtar
"""

from __future__ import division, print_function

import argparse
import json
import logging
import math
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import rhalphalib as rl
from utils import blindBins, get_template, labels, samples, shape_to_num, sigs

rl.ParametericSample.PreferRooParametricHist = False
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


from dataclasses import dataclass


@dataclass
class ShapeVar:

    """For storing and calculating info about variables used in fit"""

    name: str = None
    bins: np.ndarray = None  # bin edges
    order: int = None  # TF order

    def __post_init__(self):
        # use bin centers for polynomial fit
        self.pts = self.bins[:-1] + 0.5 * np.diff(self.bins)
        # scale to be between [0, 1]
        self.scaled = (self.pts - self.bins[0]) / (self.bins[-1] - self.bins[0])


def systs_not_from_parquets(years, LUMI, full_lumi):
    """
    Define systematics that are NOT stored in the parquets

    Returns
        systs_dict [dict]:        keys are systematics; values are rl.NuisanceParameters
        systs_dict_values [dict]: keys are same as above; values are tuples (up, down) and if (up, None) then down=up
    """

    systs_dict, systs_dict_values = {}, {}

    # branching ratio systematics
    systs_dict["BR_hww"] = rl.NuisanceParameter("BR_hww", "lnN")
    systs_dict_values["BR_hww"] = (1.0153, 0.9848)

    # lumi systematics
    if "2016" in years:
        systs_dict["lumi_13TeV_2016"] = rl.NuisanceParameter("CMS_lumi_13TeV_2016", "lnN")
        systs_dict_values["lumi_13TeV_2016"] = (1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi), None)
    if "2017" in years:
        systs_dict["lumi_13TeV_2017"] = rl.NuisanceParameter("CMS_lumi_13TeV_2017", "lnN")
        systs_dict_values["lumi_13TeV_2017"] = (1.02 ** (LUMI["2017"] / full_lumi), None)

    if "2018" in years:
        systs_dict["lumi_13TeV_2018"] = rl.NuisanceParameter("CMS_lumi_13TeV_2018", "lnN")
        systs_dict_values["lumi_13TeV_2018"] = (1.015 ** (LUMI["2018"] / full_lumi), None)

    if len(years) == 4:
        systs_dict["lumi_13TeV_correlated"] = rl.NuisanceParameter("CMS_lumi_13TeV_corelated", "lnN")
        systs_dict_values["lumi_13TeV_correlated"] = (
            (
                (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
                * (1.009 ** (LUMI["2017"] / full_lumi))
                * (1.02 ** (LUMI["2018"] / full_lumi))
            ),
            None,
        )

        systs_dict["lumi_13TeV_1718"] = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")
        systs_dict_values["lumi_13TeV_1718"] = (
            (1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi)),
            None,
        )

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
            "weight_pileup": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PU_{year}", "shape"),
            "weight_isolation": rl.NuisanceParameter(f"CMS_iso_{year}", "lnN"),
            "weight_id": rl.NuisanceParameter(f"CMS_id_{year}", "lnN"),
            "weight_reco_ele": rl.NuisanceParameter("CMS_reconstruction_ele", "lnN"),
            "weight_L1Prefiring": rl.NuisanceParameter(f"CMS_L1Prefiring_{year}", "lnN"),
            "weight_trigger_ele": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_trigger_{year}_ele", "lnN"),
            "weight_trigger_iso_mu": rl.NuisanceParameter(f"CMS_mu_trigger_iso_{year}_mu", "lnN"),
            "weight_trigger_noniso_mu": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_{year}_mu", "lnN"),
        },
        # signal
        "ggF": {
            "weight_aS_weight": rl.NuisanceParameter("aS_Higgs_ggF", "lnN"),
            "weight_UEPS_FSR": rl.NuisanceParameter("UEPS_FSR_ggF", "shape"),
            "weight_UEPS_ISR": rl.NuisanceParameter("UEPS_ISR_ggF", "shape"),
            "weight_PDF_weight": rl.NuisanceParameter("pdf_Higgs_ggF", "lnN"),
            "weight_PDFaS_weight": rl.NuisanceParameter("pdfAS_Higgs_ggF", "lnN"),
            "weight_scalevar_3pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_3_ggF_{year}", "lnN"),
            "weight_scalevar_7pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_7_ggF_{year}", "lnN"),
        },
        "VBF": {
            "weight_aS_weight": rl.NuisanceParameter("aS_Higgs_VBF", "lnN"),
            "weight_UEPS_FSR": rl.NuisanceParameter("UEPS_FSR_VBF", "shape"),
            "weight_UEPS_ISR": rl.NuisanceParameter("UEPS_ISR_VBF", "shape"),
            "weight_PDF_weight": rl.NuisanceParameter("pdf_Higgs_VBF", "lnN"),
            "weight_PDFaS_weight": rl.NuisanceParameter("pdfAS_Higgs_VBF", "lnN"),
            "weight_scalevar_3pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_3_VBF_{year}", "lnN"),
            "weight_scalevar_7pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_7_VBF_{year}", "lnN"),
        },
        "VH": {},
        "ttH": {},
        # bkgs
        "TTbar": {},
        "WJetsLNu": {
            "weight_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{year}", "lnN"),
            "weight_d2K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{year}", "lnN"),
            "weight_d3K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{year}", "lnN"),
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{year}", "lnN"),
            "weight_W_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{year}", "lnN"),
            "weight_W_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{year}", "lnN"),
        },
        "SingleTop": {},
        "DYJets": {
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{year}", "lnN"),
            "weight_Z_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{year}", "lnN"),
            "weight_Z_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{year}", "lnN"),
        },
    }

    return systs_from_parquets


def rhalphabet(hists_templates, years, channels, blind_samples, blind_region, qcd_estimation):
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(channels)

    # get the LUMI covered in the templates
    full_lumi = 0
    for year_ in years:
        full_lumi += LUMI[year_]

    # define the systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, LUMI, full_lumi)
    sys_from_parquets = systs_from_parquets(years)

    categories = list(hists_templates["pass"].axes["Category"])
    shape_var = ShapeVar(
        name=hists_templates["pass"].axes["mass_observable"].name,
        bins=hists_templates["pass"].axes["mass_observable"].edges,
        order=2,  # TODO: make the order of the polynomial configurable
    )
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    model = rl.Model("testModel")

    # fill datacard with systematics and rates
    for blind in [True, False]:
        for category in categories:
            for region in ["pass", "fail"]:
                if blind:
                    h = blindBins(hists_templates[region], blind_region, blind_samples)
                    ChName = f"{region}{category}Blinded"
                else:
                    h = hists_templates[region]
                    ChName = f"{region}{category}"

                ch = rl.Channel(ChName)
                model.addChannel(ch)

                for sName in samples:
                    templ = get_template(h, sName, category)
                    stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
                    sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

                    # SYSTEMATICS NOT FROM PARQUETS
                    for sys_name, sys_value in systs_dict.items():
                        if systs_dict_values[sys_name][1] is None:  # if up and down are the same
                            sample.setParamEffect(sys_value, systs_dict_values[sys_name][0])
                        else:
                            sample.setParamEffect(sys_value, systs_dict_values[sys_name][0], systs_dict_values[sys_name][1])

                    # SYSTEMATICS FROM PARQUETS
                    for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                        for sys_name, sys_value in sys_from_parquets[syst_on_sample].items():
                            # if "L1Prefiring" not in sys_name:
                            #     continue

                            # print(sName, sys_value, category, region)

                            syst_up = h[{"Sample": sName, "Category": category, "Systematic": sys_name + "Up"}].values()
                            syst_do = h[{"Sample": sName, "Category": category, "Systematic": sys_name + "Down"}].values()
                            nominal = h[{"Sample": sName, "Category": category, "Systematic": "nominal"}].values()

                            if sys_value.combinePrior == "lnN":
                                eff_up = shape_to_num(syst_up, nominal)
                                eff_do = shape_to_num(syst_do, nominal)

                                # if (math.isclose(1, eff_up, rel_tol=1e-4)) & (
                                #     math.isclose(1, eff_do, rel_tol=1e-4)
                                # ):  # leave it as '-'
                                #     continue

                                if math.isclose(eff_up, eff_do, rel_tol=1e-2):  # if up and down are the same
                                    sample.setParamEffect(sys_value, max(eff_up, eff_do))
                                else:
                                    sample.setParamEffect(sys_value, max(eff_up, eff_do), min(eff_up, eff_do))

                            else:
                                sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))

                    ch.addSample(sample)

                # add data
                data_obs = get_template(h, "Data", category)
                ch.setObservation(data_obs)

            if qcd_estimation:  # qcd data-driven estimation per category
                if blind:
                    h_pass = blindBins(hists_templates["pass"], blind_region, blind_samples)
                    h_fail = blindBins(hists_templates["fail"], blind_region, blind_samples)
                    passChName = f"pass{category}Blinded"
                    failChName = f"fail{category}Blinded"
                else:
                    h_pass = hists_templates["pass"]
                    h_fail = hists_templates["fail"]
                    passChName = f"pass{category}"
                    failChName = f"fail{category}"

                # get the transfer factor
                qcd_eff = (
                    h_pass[{"Category": category, "Sample": "QCD", "Systematic": "nominal"}].sum()
                    / h_fail[{"Category": category, "Sample": "QCD", "Systematic": "nominal"}].sum()
                )

                # transfer factor
                tf_dataResidual = rl.BasisPoly(
                    f"{CMS_PARAMS_LABEL}_tf_dataResidual",
                    (shape_var.order,),
                    [shape_var.name],
                    basis="Bernstein",
                    limits=(-20, 20),
                    # square_params=True,
                )
                tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
                tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

                # qcd params
                qcd_params = np.array(
                    [rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0) for i in range(m_obs.nbins)]
                )

                logging.info(f"setting transfer factor for pass region {passChName}, fail region {failChName}")

                failCh = model[failChName]
                passCh = model[passChName]

                # sideband fail
                # was integer, and numpy complained about subtracting float from it
                initial_qcd = failCh.getObservation().astype(float)
                for sample in failCh:
                    if sample.sampletype == rl.Sample.SIGNAL:
                        continue
                    # logging.info(f"subtracting {sample._name} from qcd")
                    initial_qcd -= sample.getExpectation(nominal=True)

                if np.any(initial_qcd < 0.0):
                    # raise ValueError("initial_qcd negative for some bins..", initial_qcd)
                    logging.warning(f"initial_qcd negative for some bins... {initial_qcd}")
                    initial_qcd[initial_qcd < 0] = 0

                # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
                # will result in qcdparams errors ~±1
                # but because qcd is poorly modelled we're scaling sigma scale

                sigmascale = 10  # to scale the deviation from initial
                scaled_params = initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params

                # add samples
                fail_qcd = rl.ParametericSample(
                    f"{failChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                    rl.Sample.BACKGROUND,
                    m_obs,
                    scaled_params,
                )
                failCh.addSample(fail_qcd)

                pass_qcd = rl.TransferFactorSample(
                    f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                    rl.Sample.BACKGROUND,
                    tf_params_pass,
                    fail_qcd,
                )
                passCh.addSample(pass_qcd)

    return model


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if len(args.samples_to_blind) >= 1:
        blind_samples = args.samples_to_blind.split(",")
    else:
        blind_samples = []

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "rb") as f:
        hists_templates = pkl.load(f)

    model = rhalphabet(
        hists_templates,
        years,
        channels,
        blind_samples=blind_samples,  # default is [] which means blind all samples
        blind_region=[80, 160],
        qcd_estimation=True,
    )

    with open(f"{args.outdir}/model_{save_as}.pkl", "wb") as fout:
        pkl.dump(model, fout, protocol=2)


if __name__ == "__main__":
    # e.g.
    # python create_datacard.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="name of template directory")
    parser.add_argument(
        "--samples_to_blind", dest="samples_to_blind", default="", help="samples to blind separated by commas"
    )

    args = parser.parse_args()

    main(args)
