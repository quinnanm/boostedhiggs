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
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import rhalphalib as rl
from utils import blindBins, get_template, labels, samples, shape_to_num, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"

from dataclasses import dataclass

from systematics import systs_from_parquets, systs_not_from_parquets


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


def create_datacard(hists_templates, years, channels, blind_samples, blind_region, wjets_estimation, top_estimation):
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(channels)

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

    shape_var = ShapeVar(
        name=hists_templates.axes["mass_observable"].name,
        bins=hists_templates.axes["mass_observable"].edges,
        order=2,  # TODO: make the order of the polynomial configurable
    )
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    model = rl.Model("testModel")

    # topfail, toppass = {}, {}
    # for category in regions:
    #     topfail[category], toppass[category] = {}, {}

    regions = [
        "SR1VBF",
        "SR1ggpt300to450",
        "SR1ggFpt450toInf",
        "SR1VBFBlinded",
        "SR1ggpt300to450Blinded",
        "SR1ggFpt450toInfBlinded",
        "SR2",
        "SR2Blinded",
        "WJetsCR",
        "WJetsCRBlinded",
        "TopCR",
    ]

    # fill datacard with systematics and rates
    for region in regions:
        if wjets_estimation and (region != "TopCR"):
            Samples = samples.copy()
            Samples.remove("WJetsLNu")
            Samples.remove("QCD")
        else:
            Samples = samples.copy()

        if "Blinded" in region:
            h = blindBins(hists_templates.copy(), blind_region, blind_samples)
            ChName = f"{region}"

            region = region.replace("Blinded", "")
        else:
            h = hists_templates.copy()
            ChName = f"{region}"

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
                    # print(sName, sys_value, category, region)

                    syst_up = h[{"Sample": sName, "Region": region, "Systematic": sys_name + "Up"}].values()
                    syst_do = h[{"Sample": sName, "Region": region, "Systematic": sys_name + "Down"}].values()
                    nominal = h[{"Sample": sName, "Region": region, "Systematic": "nominal"}].values()

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
        data_obs = get_template(h, "Data", region)
        ch.setObservation(data_obs)

        # # get the relevant channels for wjets estimation in pass region
        # if "TopCR" in ChName:
        #     topfail = ch["ttbar"]
        # elif "SR1Blinded" in ChName:
        #     toppass = ch["ttbar"]

    if wjets_estimation:  # data-driven estimation per category
        for region in regions:
            if "SR" not in region:
                continue

            if "Blinded" in region:
                cr = "WJetsCRBlinded"
            else:
                cr = "WJetsCR"

            rhalphabet(
                model,
                hists_templates,
                m_obs,
                shape_var,
                blind_region,
                blind_samples,
                from_region=cr,
                to_region=region,
            )

    # if top_estimation:
    #     topnormSF = rl.IndependentParameter(f"topnormSF_{year}", 1.0, -50, 50)

    #     # seperate rate of process by taking into account normalization (how well it fits data/mc in one region)
    #     # from mistag efficiency (i.e. tagger)
    #     # 2 indep dof: we don't - we choose (for now) one parameter on the overall normalization of wjets
    #     # we reparametrize both as: one normalization (both equally up and down) + effSF (one that is asymetric -
    #     # if increases, will increase in pass and decrease in fail)
    #     # for now just use normalization and see data/mc
    #     topfail.setParamEffect(topnormSF, 1 * topnormSF)
    #     toppass.setParamEffect(topnormSF, 1 * topnormSF)

    return model


def rhalphabet(model, hists_templates, m_obs, shape_var, blind_region, blind_samples, from_region="fail", to_region="pass"):
    if "Blinded" in from_region:
        assert "Blinded" in to_region
        h_fail = blindBins(hists_templates.copy(), blind_region, blind_samples)
        h_pass = blindBins(hists_templates.copy(), blind_region, blind_samples)

        failChName = f"{from_region}"
        passChName = f"{to_region}"

        to_region = to_region.replace("Blinded", "")
        from_region = from_region.replace("Blinded", "")

    if "Blinded" not in from_region:
        assert "Blinded" not in to_region
        h_fail = hists_templates.copy()
        h_pass = hists_templates.copy()

        failChName = f"{from_region}"
        passChName = f"{to_region}"

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

    # get the transfer factor
    num = (
        h_pass[{"Region": to_region, "Sample": "QCD", "Systematic": "nominal"}].sum().value
        + h_pass[{"Region": to_region, "Sample": "WJetsLNu", "Systematic": "nominal"}].sum().value
    )

    den = (
        h_fail[{"Region": from_region, "Sample": "QCD", "Systematic": "nominal"}].sum().value
        + h_fail[{"Region": from_region, "Sample": "WJetsLNu", "Systematic": "nominal"}].sum().value
    )

    qcd_eff = num / den

    # qcd params
    qcd_params = np.array(
        [rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0) for i in range(m_obs.nbins)]
    )

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

    logging.info(f"setting transfer factor for region {passChName}, from region {failChName}")

    passCh = model[passChName]

    pass_qcd = rl.TransferFactorSample(
        f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
        rl.Sample.BACKGROUND,
        tf_params_pass,
        fail_qcd,
    )
    passCh.addSample(pass_qcd)


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

    model = create_datacard(
        hists_templates,
        years,
        channels,
        blind_samples=blind_samples,  # default is [] which means blind all samples
        blind_region=[90, 150],
        wjets_estimation=True,
        top_estimation=False,
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
