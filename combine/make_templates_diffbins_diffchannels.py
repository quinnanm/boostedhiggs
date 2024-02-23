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
    "ele": [
        "weight_btagSFlightCorrelated",
        "weight_btagSFlight2016",
        "weight_btagSFbc2016",
        "weight_btagSFlight2016APV",
        "weight_btagSFbc2016APV",
        "weight_btagSFlight2017",
        "weight_btagSFbc2017",
        "weight_btagSFlight2018",
        "weight_btagSFbc2018",
        "weight_ele_pileup",
        "weight_ele_pileupIDSFDown",
        "weight_ele_isolation_electron",
        "weight_ele_id_electron",
        "weight_ele_reco_electron",
        "weight_ele_L1Prefiring",
        "weight_ele_trigger_electron",
        "weight_ele_PSFSR_weight",
        "weight_ele_PSISR_weight",
        "weight_ele_d1kappa_EW",
        "weight_ele_d1K_NLO",
        "weight_ele_d2K_NLO",
        "weight_ele_d3K_NLO",
        "weight_ele_W_d2kappa_EW",
        "weight_ele_W_d3kappa_EW",
        "weight_ele_Z_d2kappa_EW",
        "weight_ele_Z_d3kappa_EW",
    ],
    "mu": [
        "weight_btagSFbcCorrelated",
        "weight_btagSFlight2016",
        "weight_btagSFbc2016",
        "weight_btagSFlight2016APV",
        "weight_btagSFbc2016APV",
        "weight_btagSFlight2017",
        "weight_btagSFbc2017",
        "weight_btagSFlight2018",
        "weight_btagSFbc2018",
        "weight_mu_pileup",
        "weight_mu_pileupIDSFDown",
        "weight_mu_isolation_muon",
        "weight_mu_id_muon",
        "weight_mu_L1Prefiring",
        "weight_mu_trigger_iso_muon",
        "weight_mu_trigger_noniso_muon",
        "weight_mu_PSFSR",
        "weight_mu_PSISR",
        "weight_mu_d1kappa_EW",
        "weight_mu_d1K_NLO",
        "weight_mu_d2K_NLO",
        "weight_mu_d3K_NLO",
        "weight_mu_W_d2kappa_EW",
        "weight_mu_W_d3kappa_EW",
        "weight_mu_Z_d2kappa_EW",
        "weight_mu_Z_d3kappa_EW",
    ],
}

AK8_systs = [
    "rec_higgs_mUES_up",
    "rec_higgs_mUES_down",
    "rec_higgs_mJES_up",
    "rec_higgs_mJES_down",
    "rec_higgs_mJER_up",
    "rec_higgs_mJER_down",
    # these
    "rec_higgs_mJMS_up",
    "rec_higgs_mJMS_down",
    "rec_higgs_mJMR_up",
    "rec_higgs_mJMR_down",
]


def get_templates(years, channels, samples, samples_dir, regions_sel, regions_massbins, model_path):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "QCD", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (fj_ParT_score>0.97)}`)
        model_path [str]: path to the ParT finetuned model.onnx

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (samples, systematic, ptbin, mass_observable)

    """

    # add extra selections to preselection
    presel = {
        "mu": {
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
        },
        "ele": {
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
        },
    }

    hists = {}
    for ch in channels:  # e.g. mu, ele
        for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
            hists[f"{region}_{ch}"] = hist2.Hist(
                hist2.axis.StrCategory([], name="Sample", growth=True),
                hist2.axis.StrCategory([], name="Systematic", growth=True),
                hist2.axis.Variable(
                    list(range(50, 240, regions_massbins[region])),
                    name="mass_observable",
                    label=r"Higgs reconstructed mass [GeV]",
                    overflow=True,
                ),
                storage=hist2.storage.Weight(),
            )

            for year in years:  # e.g. 2018, 2017, 2016APV, 2016

                logging.info(f"Processing year {year} and {ch} channel")

                with open("../fileset/luminosity.json") as f:
                    luminosity = json.load(f)[ch][year]

                for sample in os.listdir(samples_dir[year]):

                    if "WJetsToLNu_1J" in sample:
                        print(f"Skipping sample {sample}")
                        continue
                    if "WJetsToLNu_2J" in sample:
                        print(f"Skipping sample {sample}")
                        continue

                    if "VBFHToWWToLNuQQ_" in sample:
                        print(f"Skipping sample {sample}")
                        continue

                    # first: check if the sample is in one of combine_samples_by_name
                    sample_to_use = None
                    for key in utils.combine_samples_by_name:
                        if key in sample:
                            sample_to_use = utils.combine_samples_by_name[key]
                            break

                    # second: if not, combine under common label
                    if sample_to_use is None:
                        for key in utils.combine_samples:
                            if key in sample:
                                sample_to_use = utils.combine_samples[key]
                                break
                            else:
                                sample_to_use = sample

                    if sample_to_use not in samples:
                        continue

                    logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                    out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                    pkl_files = glob.glob(f"{out_files}/*.pkl")

                    if not parquet_files:
                        logging.info(f"No parquet file for {sample}")
                        continue

                    try:
                        data = pd.read_parquet(parquet_files)
                    except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                        continue

                    if len(data) == 0:
                        continue

                    # use hidNeurons to get the finetuned scores
                    data["fj_ParT_score_finetuned"] = utils.get_finetuned_score(data, model_path)

                    # drop hidNeurons which are not needed anymore
                    data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                    # apply selection
                    for selection in presel[ch]:
                        logging.info(f"Applying {selection} selection on {len(data)} events")
                        data = data.query(presel[ch][selection])

                    # get event_weight
                    if sample_to_use == "Data":
                        is_data = True
                    else:
                        is_data = False

                    event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)

                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(data)} events")

                    df = df.query(region_sel)

                    if not is_data:
                        W = df[f"weight_{ch}"]

                        if "bjets" in region_sel:  # add btag SF
                            W *= df["weight_btag"]

                    logging.info(f"Will fill the {region}_{ch} histogram with the remaining {len(data)} events")

                    # add nominal weight
                    if is_data:  # for data (nominal is 1)
                        nominal = np.ones_like(df["fj_pt"])
                    else:
                        nominal = W * event_weight
                    hists[f"{region}_{ch}"].fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

                    # add up/down variations
                    for weight in weights[ch]:

                        if is_data:  # for data (fill as 1 for up and down variations)
                            w = nominal

                        # retrieve UP variations for MC
                        if not is_data:
                            try:
                                w = df[f"{weight}Up"] * event_weight
                                if "btag" in weight:
                                    w *= W
                            except KeyError:
                                w = nominal

                        hists[f"{region}_{ch}"].fill(
                            Sample=sample_to_use,
                            Systematic=f"{weight}_up",
                            mass_observable=df["rec_higgs_m"],
                            weight=w,
                        )

                        # retrieve DOWN variations for MC
                        if not is_data:
                            try:
                                w = df[f"{weight}Down"] * event_weight
                                if "btag" in weight:
                                    w *= W
                            except KeyError:
                                w = nominal

                        hists[f"{region}_{ch}"].fill(
                            Sample=sample_to_use,
                            Systematic=f"{weight}_down",
                            mass_observable=df["rec_higgs_m"],
                            weight=w,
                        )

                    for rec_higgs_m_variation in AK8_systs:

                        if is_data:
                            x = "rec_higgs_m"
                        else:
                            x = rec_higgs_m_variation

                        hists[f"{region}_{ch}"].fill(
                            Sample=sample_to_use,
                            Systematic=rec_higgs_m_variation,
                            mass_observable=df[x],
                            weight=nominal,
                        )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h:
        for sample in h[region].axes["Sample"]:
            neg_bins = np.where(h[region][{"Sample": sample, "Systematic": "nominal"}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h[region].axes["Sample"]) == sample)

                for neg_bin in neg_bins:
                    h[region].view(flow=True)[sample_index, :, neg_bin + 1].value = 0
                    h[region].view(flow=True)[sample_index, :, neg_bin + 1].variance = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates_diffbins.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["regions_massbins"],
        config["model_path"],
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates_diffbins_diffchannels.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v11

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
