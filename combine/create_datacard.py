"""
Creates "combine datacards" using hist.Hist templates, and
sets up data-driven QCD background estimate ('rhalphabet' method)

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
"""

from __future__ import division, print_function

import argparse
import json
import logging
import math
import os
import pickle as pkl
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rhalphalib as rl
from systematics import systs_from_parquets, systs_not_from_parquets
from utils import blindBins, get_template, labels, samples, shape_to_num, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


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


def create_datacard(hists_templates, years, lep_channels, wjets_estimation):
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in lep_channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(lep_channels)

    if len(years) == 4:
        year = "Run2"
    else:
        year = years[0]

    # get the LUMI covered in the templates
    full_lumi = 0
    for year_ in years:
        full_lumi += LUMI[year_]

    # define the systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, LUMI, full_lumi)
    sys_from_parquets = systs_from_parquets(years)

    model = rl.Model("testModel")

    regions = [
        "SR1VBF",
        "SR1ggFpt300to450",
        "SR1ggFpt450toInf",
        "SR2",
        # "SR1VBFBlinded",
        # "SR1ggFpt300to450Blinded",
        # "SR1ggFpt450toInfBlinded",
        # "SR2Blinded",
    ]  # put the signal regions here

    regions += ["WJetsCR"]
    # regions += ["WJetsCRBlinded"]

    # fill datacard with systematics and rates
    # ChName may have "Blinded" in the string, but region does not
    for ChName in regions:
        if wjets_estimation:  # only use MC qcd and wjets for Top control region
            Samples = samples.copy()
            Samples.remove("WJetsLNu")
            Samples.remove("QCD")
        else:
            Samples = samples.copy()

        if "Blinded" in ChName:
            h = blindBins(hists_templates.copy())
            region = ChName.replace("Blinded", "")  # region will be used to get the axes of the templates
        else:
            h = hists_templates.copy()
            region = ChName

        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in Samples:
            templ = get_template(h, sName, region)
            stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

            sample.autoMCStats(lnN=True)

            # SYSTEMATICS NOT FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in systs_dict[syst_on_sample].items():
                    if systs_dict_values[syst_on_sample][sys_name][1] is None:  # if up and down are the same
                        sample.setParamEffect(sys_value, systs_dict_values[syst_on_sample][sys_name][0])
                    else:
                        sample.setParamEffect(
                            sys_value,
                            systs_dict_values[syst_on_sample][sys_name][0],
                            systs_dict_values[syst_on_sample][sys_name][1],
                        )

            # SYSTEMATICS FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in sys_from_parquets[syst_on_sample].items():
                    syst_up = h[{"Sample": sName, "Region": region, "Systematic": sys_name + "Up"}].values()
                    syst_do = h[{"Sample": sName, "Region": region, "Systematic": sys_name + "Down"}].values()
                    nominal = h[{"Sample": sName, "Region": region, "Systematic": "nominal"}].values()

                    if sys_value.combinePrior == "lnN":
                        eff_up = shape_to_num(syst_up, nominal)
                        eff_do = shape_to_num(syst_do, nominal)

                        if math.isclose(eff_up, eff_do, rel_tol=1e-2):  # if up and down are the same
                            sample.setParamEffect(sys_value, max(eff_up, eff_do))
                        else:
                            sample.setParamEffect(sys_value, max(eff_up, eff_do), min(eff_up, eff_do))

                    else:
                        sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))

            ch.addSample(sample)

        # add data
        data_obs = get_template(h, "Data", region)
        ch.setObservation(data_obs)

    if wjets_estimation:  # data-driven estimation
        failChName = "WJetsCR"
        passChNames = ["SR1VBF", "SR1ggFpt300to450", "SR1ggFpt450toInf", "SR2"]
        # passChNames = ["SR1ggFpt300to450"]
        rhalphabet(
            model,
            hists_templates,
            passChNames,
            failChName,
        )

        # failChName = "WJetsCRBlinded"
        # passChNames = ["SR1VBFBlinded", "SR1ggFpt300to450Blinded", "SR1ggFpt450toInfBlinded", "SR2Blinded"]
        # # passChNames = ["SR1ggFpt300to450Blinded"]
        # rhalphabet(
        #     model,
        #     hists_templates,
        #     passChNames,
        #     failChName,
        # )

    return model


def rhalphabet(model, hists_templates, passChNames, failChName):
    shape_var = ShapeVar(
        name=hists_templates.axes["mass_observable"].name,
        bins=hists_templates.axes["mass_observable"].edges,
        order=2,  # TODO: make the order of the polynomial configurable
    )
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    # qcd params
    qcd_params = np.array(
        [rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0) for i in range(m_obs.nbins)]
    )

    failCh = model[failChName]

    initial_qcd = failCh.getObservation().astype(float)
    if np.any(initial_qcd < 0.0):
        # raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        logging.warning(f"initial_qcd negative for some bins... {initial_qcd}")
        initial_qcd[initial_qcd < 0] = 0

    for sample in failCh:
        if sample.sampletype == rl.Sample.SIGNAL:
            continue
        # logging.info(f"subtracting {sample._name} from qcd")
        initial_qcd -= sample.getExpectation(nominal=True)

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

    for passChName in passChNames:
        logging.info(f"setting transfer factor for region {passChName}, from region {failChName}")

        if "Blinded" in failChName:
            assert "Blinded" in passChName
            h = blindBins(hists_templates.copy())

            den = (
                h[{"Region": failChName.replace("Blinded", ""), "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}]
                .sum()
                .value
            )
            num = (
                h[{"Region": passChName.replace("Blinded", ""), "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}]
                .sum()
                .value
            )

        else:
            assert "Blinded" not in passChName
            h = hists_templates.copy()

            den = h[{"Region": failChName, "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}].sum().value
            num = h[{"Region": passChName, "Sample": ["WJetsLNu", "QCD"], "Systematic": "nominal"}].sum().value

        # get the transfer factor
        qcd_eff = num / den

        # transfer factor
        tf_dataResidual = rl.BasisPoly(
            f"{CMS_PARAMS_LABEL}_tf_dataResidual_{passChName}",
            (shape_var.order,),
            [shape_var.name],
            basis="Bernstein",
            limits=(-20, 20),
            # square_params=True,   # TODO: figure why this can't be uncommented
        )
        tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
        tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

        passCh = model[passChName]
        pass_qcd = rl.TransferFactorSample(
            f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            tf_params_pass,
            fail_qcd,
        )
        passCh.addSample(pass_qcd)


def load_templates(years, lep_channels, outdir):
    # load templates
    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(lep_channels) == 1:
        save_as += f"_{lep_channels[0]}_"

    with open(f"{outdir}/hists_templates_{save_as}.pkl", "rb") as f:
        hists_templates = pkl.load(f)

    return hists_templates


def main(args):
    years = args.years.split(",")
    lep_channels = args.channels.split(",")

    hists_templates = load_templates(years, lep_channels, args.outdir)

    model = create_datacard(hists_templates, years, lep_channels, wjets_estimation=args.wjets_estimation)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="name of template directory")
    parser.add_argument("--wjets-estimation", action="store_true")

    args = parser.parse_args()

    main(args)
