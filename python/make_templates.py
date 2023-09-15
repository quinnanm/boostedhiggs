"""
Builds hist.Hist templates after adding systematics for all samples

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


weights = {
    "mu": [
        "weight_mu_btagSFlight_2017",
        "weight_mu_btagSFlight_correlated",
        "weight_mu_btagSFbc_2017",
        "weight_mu_btagSFbc_correlated",
        "weight_mu_pileup",
        "weight_mu_isolation_muon",
        "weight_mu_id_muon",
        "weight_mu_L1Prefiring",
        "weight_mu_trigger_iso_muon",
        "weight_mu_trigger_noniso_muon",
        # ggF & VBF
        "weight_mu_aS_weight",
        "weight_mu_UEPS_FSR",
        "weight_mu_UEPS_ISR",
        "weight_mu_PDF_weight",
        "weight_mu_PDFaS_weight",
        "weight_mu_scalevar_3pt",
        "weight_mu_scalevar_7pt",
        # WJetsLNu
        "weight_mu_d1K_NLO",
        "weight_mu_d2K_NLO",
        "weight_mu_d3K_NLO",
        "weight_mu_d1kappa_EW",
        "weight_mu_W_d2kappa_EW",
        "weight_mu_W_d3kappa_EW",
        # DY
        "weight_mu_Z_d2kappa_EW",
        "weight_mu_Z_d3kappa_EW",
    ],
    "ele": [
        "weight_ele_btagSFlight_2017",
        "weight_ele_btagSFlight_correlated",
        "weight_ele_btagSFbc_2017",
        "weight_ele_btagSFbc_correlated",
        "weight_ele_pileup",
        "weight_ele_isolation_electron",
        "weight_ele_id_electron",
        "weight_ele_L1Prefiring",
        "weight_ele_trigger_electron",
        "weight_ele_reco_electron",
        # ggF & VBF
        "weight_ele_UEPS_FSR",
        "weight_ele_UEPS_ISR",
        "weight_ele_PDF_weight",
        "weight_ele_PDFaS_weight",
        "weight_ele_scalevar_3pt",
        "weight_ele_scalevar_7pt",
        # WJetsLNu
        "weight_ele_d1K_NLO",
        "weight_ele_d2K_NLO",
        "weight_ele_d3K_NLO",
        "weight_ele_d1kappa_EW",
        "weight_ele_W_d2kappa_EW",
        "weight_ele_W_d3kappa_EW",
    ],
}


def get_templates(year, ch, samples_dir, samples, presel, regions_selections):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        year [str]: year to postprocess and save in the output (e.g. ["2016APV", "2016"])
        ch [str]: channel to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [dict]: points to the path of the parquets for each region (note: the year will be appended)
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)
        regions_selections [dict]: (e.g. `{"pass": ( (inclusive_score>0.99) & (n_bjets_M < 2) )}`)

    Returns
        a dict() object hists[region] that contains histograms

    """

    # get lumi
    luminosity = 0
    with open("../fileset/luminosity.json") as f:
        luminosity += json.load(f)[ch][year]

    bins = {
        "fj_pt": [200, 300, 450, 650, 2000],
        "rec_higgs_m": list(range(40, 240, 21)),
    }

    hists = {}
    for region in regions_selections:
        hists[region] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            hist2.axis.StrCategory([], name="systematic", growth=True),
            hist2.axis.Variable(bins["fj_pt"], name="fj_pt", label=r"Jet $p_T$ [GeV]", overflow=True),
            hist2.axis.Variable(
                bins["rec_higgs_m"], name="rec_higgs_m", label=r"Higgs reconstructed mass [GeV]", overflow=True
            ),
        )

    logging.info(f"Processing year {year} and {ch} channel")

    for region in regions_selections:
        for sample in os.listdir(samples_dir[region] + year):
            # get a combined label to combine samples of the same process
            for key in utils.combine_samples:
                if key in sample:
                    sample_to_use = utils.combine_samples[key]
                    break
                else:
                    sample_to_use = sample

            if sample_to_use not in samples:
                logging.info(f"ATTENTION: {sample} will be skipped")
                continue

            logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

            out_files = f"{samples_dir[region] + year}/{sample}/outfiles/"
            parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            pkl_files = glob.glob(f"{out_files}/*.pkl")

            if not parquet_files:
                logging.info(f"No parquet file for {sample}")
                continue

            try:
                data = pd.read_parquet(parquet_files)

            except pyarrow.lib.ArrowInvalid:
                # empty parquet because no event passed selection
                continue

            if len(data) == 0:
                continue

            # apply selection
            for selection in presel[ch]:
                data = data.query(presel[ch][selection])

            # get event_weight
            if sample_to_use != "Data":
                event_weight = utils.get_xsecweight(pkl_files, year, sample, True, luminosity)
            else:
                data[f"weight_{ch}"] = 1  # not stored for data
                event_weight = 1

            # add tagger scores
            data["inclusive_score"] = data["fj_ParT_all_score"]

            # fill histograms
            data = data.query(regions_selections[region])

            # Nominal weight
            nominal = data[f"weight_{ch}"] * event_weight

            hists[region].fill(
                samples=sample_to_use,
                systematic="nominal",
                fj_pt=data["fj_pt"],
                rec_higgs_m=data["rec_higgs_m"],
                weight=nominal,
            )

            for weight in weights[ch]:
                # Up weight
                try:
                    syst = data[f"{weight}Up"] * event_weight

                except KeyError:
                    syst = np.ones_like(data["fj_pt"])

                hists[region].fill(
                    samples=sample_to_use,
                    systematic=f"{weight}Up",
                    fj_pt=data["fj_pt"],
                    rec_higgs_m=data["rec_higgs_m"],
                    weight=syst,
                )

                # Down weight
                try:
                    syst = data[f"{weight}Down"] * event_weight

                except KeyError:
                    syst = np.ones_like(data["fj_pt"])

                hists[region].fill(
                    samples=sample_to_use,
                    systematic=f"{weight}Down",
                    fj_pt=data["fj_pt"],
                    rec_higgs_m=data["rec_higgs_m"],
                    weight=syst,
                )

    logging.info(hists)

    return hists


def main(args):
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    if not os.path.exists(f"{args.outpath}/{args.tag}"):
        os.makedirs(f"{args.outpath}/{args.tag}")

    for ch in channels:
        os.system(f"cp config_make_templates.yaml {args.outpath}/{args.tag}/hists_templates_{args.year}_{ch}_config.yaml")

        # load config from yaml
        with open("config_make_templates.yaml", "r") as stream:
            config = yaml.safe_load(stream)

        hists = get_templates(
            args.year,
            ch,
            config["samples_dir"],
            config["samples"],
            config["presel"],
            config["regions_selections"],
        )

        with open(f"{args.outpath}/{args.tag}/hists_templates_{args.year}_{ch}.pkl", "wb") as fp:
            pkl.dump(hists, fp)

    # # dump the templates of each region in a rootfile
    # if not os.path.exists(f"{args.outpath}/hists_templates"):
    #     os.makedirs(f"{args.outpath}/hists_templates")

    # for region in hists.keys():
    #     file = uproot.recreate(f"{args.outpath}/hists_templates/{region}.root")

    #     for sample in hists[region].axes["samples"]:
    #         if sample == "Data":
    #             file[f"{region}/data_obs"] = hists[region][{"fj_pt": sum, "samples": sample}]
    #             continue
    #         file[f"{region}/{sample}"] = hists[region][{"fj_pt": sum, "samples": sample}]


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --year 2017 --channels mu,ele --tag test

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", help="year", choices=["2016APV", "2016", "2017", "2018"])
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outpath", dest="outpath", default="../combine/templates/", type=str, help="path of the output")
    parser.add_argument("--tag", dest="tag", default="test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
