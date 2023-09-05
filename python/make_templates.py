#!/usr/bin/python

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
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

"""
TODO:
  Create datacard script: make_cards.py
  e.g. similar to
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
"""

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
        # HWW & VBF
        "weight_mu_aS_weight",
        "weight_mu_UEPS_FSR",
        "weight_mu_UEPS_ISR",
        "weight_mu_PDF_weight",
        "weight_mu_PDFaS_weight",
        "weight_mu_scalevar_3pt",
        "weight_mu_scalevar_7pt",
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
        # WJetsLNu
        "weight_ele_d1K_NLO",
        "weight_ele_d2K_NLO",
        "weight_ele_d3K_NLO",
        "weight_ele_d1kappa_EW",
        "weight_ele_W_d2kappa_EW",
        "weight_ele_W_d3kappa_EW",
        # HWW
        "weight_ele_UEPS_FSR",
        "weight_ele_UEPS_ISR",
        "weight_ele_PDF_weight",
        "weight_ele_PDFaS_weight",
        "weight_ele_scalevar_3pt",
        "weight_ele_scalevar_7pt",
    ],
}


def get_templates(years, channels, samples_dir, samples, presel, regions_selections):
    """
    Postprocess the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets (note: the year will be appended to the string)
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)
        regions_selections [dict]: (e.g. `{"pass": ( (inclusive_score>0.99) & (n_bjets_M < 2) )}`)

    Returns
        a dict() object hists[region] that contains histograms

    """

    bins = {
        "fj_pt": [200, 300, 450, 650, 1200],
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

    for year in years:
        logging.info(f"Processing year {year}")
        for ch in channels:
            # get lumi
            luminosity = 0
            with open("../fileset/luminosity.json") as f:
                luminosity += json.load(f)[ch][year]

            condor_dir = os.listdir(samples_dir + year)
            for sample in condor_dir:
                if sample == "DYJetsToLL_M-10to50":
                    continue  # because tagger didnt run for it

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

                out_files = f"{samples_dir + year}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                data = pd.read_parquet(parquet_files)
                if len(data) == 0:
                    continue

                # apply selection
                for selection in presel[ch]:
                    data = data.query(presel[ch][selection])

                data["inclusive_score"] = data["fj_ParT_inclusive_score"]

                if sample_to_use == "Data":
                    data[f"weight_{ch}"] = 1

                # fill histograms
                for region in regions_selections:
                    data1 = data.copy()  # get fresh copy of the data to apply selections on
                    data1 = data1.query(regions_selections[region])

                    # Nominal weight
                    hists[region].fill(
                        samples=sample_to_use,
                        systematic="Nominal",
                        fj_pt=data1["fj_pt"],
                        rec_higgs_m=data1["rec_higgs_m"],
                        weight=data1[f"weight_{ch}"],
                    )

                    # Up weight
                    for weight in weights[ch]:
                        try:
                            syst = data1[f"{weight}Up"]

                        except KeyError:
                            logging.info(f"can't find {weight}Up systematic for {sample} sample")
                            syst = np.zeros_like(data1["fj_pt"])

                        hists[region].fill(
                            samples=sample_to_use,
                            systematic=f"{weight}Up",
                            fj_pt=data1["fj_pt"],
                            rec_higgs_m=data1["rec_higgs_m"],
                            weight=syst,
                        )

                        # Down weight
                        try:
                            syst = data1[f"{weight}Down"]

                        except KeyError:
                            logging.info(f"can't find {weight}Down systematic for {sample} sample")
                            syst = np.zeros_like(data1["fj_pt"])

                        hists[region].fill(
                            samples=sample_to_use,
                            systematic=f"{weight}Down",
                            fj_pt=data1["fj_pt"],
                            rec_higgs_m=data1["rec_higgs_m"],
                            weight=syst,
                        )

    logging.info(hists)

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp make_templates_config.yaml {args.outpath}/")

    # load config from yaml
    with open("make_templates_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    hists = get_templates(
        years,
        channels,
        args.samples_dir,
        config["samples"],
        config["presel"],
        config["regions_selections"],
    )

    with open(f"{args.outpath}/hists_templates.pkl", "wb") as fp:
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
    # python make_templates.py --years 2017 --channels ele,mu

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="ele", help="channels separated by commas")
    parser.add_argument("--samples_dir", dest="samples_dir", default="../eos/Jul21_", type=str, help="path to parquets")
    parser.add_argument(
        "--outpath", dest="outpath", default="/Users/fmokhtar/Desktop/hww/test", type=str, help="path of the output"
    )
    args = parser.parse_args()

    main(args)
