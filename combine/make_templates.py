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


# ("key", "value"): the "key" is the common naming (to commonalize over both channels)
weights = {
    "weight_pdf_acceptance": {},
    "weight_qcd_scale": {},
    # common for all samples
    "weight_btagSFlightCorrelated": {"mu": "weight_btagSFlightCorrelated", "ele": "weight_btagSFlightCorrelated"},
    "weight_btagSFbcCorrelated": {"mu": "weight_btagSFbcCorrelated", "ele": "weight_btagSFbcCorrelated"},
    "weight_btagSFlight2016": {"mu": "weight_btagSFlight2016", "ele": "weight_btagSFlight2016"},
    "weight_btagSFbc2016": {"mu": "weight_btagSFbc2016", "ele": "weight_btagSFbc2016"},
    "weight_btagSFlight2016APV": {"mu": "weight_btagSFlight2016APV", "ele": "weight_btagSFlight2016APV"},
    "weight_btagSFbc2016APV": {"mu": "weight_btagSFbc2016APV", "ele": "weight_btagSFbc2016APV"},
    "weight_btagSFlight2017": {"mu": "weight_btagSFlight2017", "ele": "weight_btagSFlight2017"},
    "weight_btagSFbc2017": {"mu": "weight_btagSFbc2017", "ele": "weight_btagSFbc2017"},
    "weight_btagSFlight2018": {"mu": "weight_btagSFlight2018", "ele": "weight_btagSFlight2018"},
    "weight_btagSFbc2018": {"mu": "weight_btagSFbc2018", "ele": "weight_btagSFbc2018"},
    "weight_pileup": {"mu": "weight_mu_pileup", "ele": "weight_ele_pileup"},
    "weight_pileupIDSF": {"mu": "weight_mu_pileupIDSFDown", "ele": "weight_ele_pileupIDSFDown"},
    "weight_isolation_mu": {"mu": "weight_mu_isolation_muon", "ele": ""},
    "weight_isolation_ele": {"mu": "", "ele": "weight_ele_isolation_electron"},
    "weight_id_mu": {"mu": "weight_mu_id_muon", "ele": ""},
    "weight_id_ele": {"mu": "", "ele": "weight_ele_id_electron"},
    "weight_reco_ele": {"mu": "", "ele": "weight_ele_reco_electron"},
    "weight_L1Prefiring": {"mu": "weight_mu_L1Prefiring", "ele": "weight_ele_L1Prefiring"},
    "weight_trigger_ele": {"mu": "", "ele": "weight_ele_trigger_electron"},
    "weight_trigger_iso_mu": {"mu": "weight_mu_trigger_iso_muon", "ele": ""},
    "weight_trigger_noniso_mu": {"mu": "weight_mu_trigger_noniso_muon", "ele": ""},
    # ggF & VBF
    "weight_PSFSR": {"mu": "weight_mu_PSFSR", "ele": "weight_ele_PSFSR_weight"},
    "weight_PSISR": {"mu": "weight_mu_PSISR", "ele": "weight_ele_PSISR_weight"},
    # WJetsLNu & DY
    "weight_d1kappa_EW": {"mu": "weight_mu_d1kappa_EW", "ele": "weight_ele_d1kappa_EW"},
    # WJetsLNu
    "weight_d1K_NLO": {"mu": "weight_mu_d1K_NLO", "ele": "weight_ele_d1K_NLO"},
    "weight_d2K_NLO": {"mu": "weight_mu_d2K_NLO", "ele": "weight_ele_d2K_NLO"},
    "weight_d3K_NLO": {"mu": "weight_mu_d3K_NLO", "ele": "weight_ele_d3K_NLO"},
    "weight_W_d2kappa_EW": {"mu": "weight_mu_W_d2kappa_EW", "ele": "weight_ele_W_d2kappa_EW"},
    "weight_W_d3kappa_EW": {"mu": "weight_mu_W_d3kappa_EW", "ele": "weight_ele_W_d3kappa_EW"},
    # DY
    "weight_Z_d2kappa_EW": {"mu": "weight_mu_Z_d2kappa_EW", "ele": "weight_ele_Z_d2kappa_EW"},
    "weight_Z_d3kappa_EW": {"mu": "weight_mu_Z_d3kappa_EW", "ele": "weight_ele_Z_d3kappa_EW"},
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

# shape_weights = {
#     # "fj_pt": [
#     #     "fj_ptJES_up",
#     #     "fj_ptJES_down",
#     #     "fj_ptJER_up",
#     #     "fj_ptJER_down",
#     # ],
#     # "fj_mass": [
#     #     "fj_massJMS_up",
#     #     "fj_massJMS_down",
#     #     "fj_massJMR_up",
#     #     "fj_massJMR_down",
#     # ],
#     # "mjj": [
#     #     "mjjJES_up",
#     #     "mjjJES_down",
#     #     "mjjJER_up",
#     #     "mjjJER_down",
#     # ],
#     "rec_higgs_m": [
#         "rec_higgs_mUES_up",
#         "rec_higgs_mUES_down",
#         "rec_higgs_mJES_up",
#         "rec_higgs_mJES_down",
#         "rec_higgs_mJER_up",
#         "rec_higgs_mJER_down",


#         "rec_higgs_mJMS_up",
#         "rec_higgs_mJMS_down",
#         "rec_higgs_mJMR_up",
#         "rec_higgs_mJMR_down",
#     ],
#     # "rec_higgs_pt": [
#     #     "rec_higgs_ptUES_up",
#     #     "rec_higgs_ptUES_down",
#     #     "rec_higgs_ptJES_up",
#     #     "rec_higgs_ptJES_down",
#     #     "rec_higgs_ptJER_up",
#     #     "rec_higgs_ptJER_down",
#     #     "rec_higgs_ptJMS_up",
#     #     "rec_higgs_ptJMS_down",
#     #     "rec_higgs_ptJMR_up",
#     #     "rec_higgs_ptJMR_down",
#     # ],
# }


def get_common_name(sample):
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
    return sample_to_use


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path):
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
            "tagger>0.80": "fj_ParT_score_finetuned>0.80",
            # "msoftdrop": "fj_mass>40",
            # "met": "met_pt>35",
        },
        "ele": {
            "tagger>0.80": "fj_ParT_score_finetuned>0.80",
            # "msoftdrop": "fj_mass>40",
            # "met": "met_pt>55",
            "lepmiso": "(lep_pt<120) | ( (lep_pt>120) & (lep_misolation<0.2))",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            list(range(55, 255, mass_binning)),
            name="mass_observable",
            label=r"Higgs reconstructed mass [GeV]",
            overflow=True,
        ),
        storage=hist2.storage.Weight(),
    )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

                if "VBFHToWWToLNuQQ_" in sample:
                    print(f"Skipping sample {sample}")
                    continue

                sample_to_use = get_common_name(sample)

                if sample_to_use not in samples:
                    continue

                if sample_to_use == "Data":
                    is_data = True
                else:
                    is_data = False

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
                event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(data)} events")

                    df = df.query(region_sel)

                    # get the nominal weight
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * event_weight

                        # if "bjets" in region_sel:  # add btag SF
                        #     nominal *= df["weight_btag"]

                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

                    # add Up/Down variations
                    for weight in weights:

                        if weight == "weight_pdf_acceptance":
                            if sample_to_use in ["ggF", "VBF", "VH", "ZH"]:

                                pdfweights = df.loc[:, df.columns.str.contains("pdf")]

                                abs_unc = np.linalg.norm((pdfweights.values - np.array(nominal).reshape(-1, 1)), axis=1)
                                # cap at 100% uncertainty
                                rel_unc = np.clip(abs_unc / nominal, 0, 1)
                                shape_up = nominal * (1 + rel_unc)
                                shape_down = nominal * (1 - rel_unc)

                                shape_down[shape_down < 0] = 0.0001
                            else:
                                shape_up = nominal
                                shape_down = nominal

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic="weight_pdf_up",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_up,
                            )

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic="weight_pdf_down",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_down,
                            )

                        elif weight == "weight_qcd_scale":
                            if sample_to_use in ["ggF", "VBF", "VH", "ZH", "WJetsLNu", "TTbar"]:
                                scaleweights = df.loc[:, df.columns.str.contains("weight_scale")]

                                shape_up = np.max(scaleweights.values, axis=1) * nominal
                                shape_down = np.min(scaleweights.values, axis=1) * nominal

                            else:
                                shape_up = nominal
                                shape_down = nominal

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic="weight_scale_up",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_up,
                            )

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic="weight_scale_down",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_down,
                            )

                        else:  # all the other weights
                            if is_data:  # for data (fill as 1 for up and down variations)
                                shape_up = nominal
                                shape_down = nominal

                            # retrieve Up variations for MC
                            else:
                                try:
                                    shape_up = df[f"{weights[weight][ch]}Up"] * event_weight
                                    if "btag" in weight:
                                        shape_up *= df["weight_btag"]
                                except KeyError:
                                    shape_up = nominal

                                try:
                                    shape_down = df[f"{weights[weight][ch]}Down"] * event_weight
                                    if "btag" in weight:
                                        shape_down *= df["weight_btag"]
                                except KeyError:
                                    shape_down = nominal

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic=f"{weight}_up",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_up,
                            )

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic=f"{weight}_down",
                                Region=region,
                                mass_observable=df["rec_higgs_m"],
                                weight=shape_down,
                            )

                    for rec_higgs_m_variation in AK8_systs:

                        if is_data:
                            x = "rec_higgs_m"
                        else:
                            x = rec_higgs_m_variation

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=rec_higgs_m_variation,
                            Region=region,
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
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 0
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years, channels, config["samples"], config["samples_dir"], config["regions_sel"], config["model_path"]
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
