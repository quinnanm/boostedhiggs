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


def get_templates(years, channels, samples, samples_dir, model_path):
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

    # TODO: fix the variables depeding onb what yo want
    vars_ = ["fj_pt", "lep_pt"]
    hists = {}

    for var_ in vars_:
        hists[var_] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            hist2.axis.Variable(list(range(50, 240, 10)), name=var_, label=var_, overflow=True),
            storage=hist2.storage.Weight(),
        )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

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
                    data = pd.read_parquet(
                        parquet_files
                    )  # TODO: if you run into memory issues, maybe load one parquet at a time

                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["fj_ParT_score_finetuned"] = utils.get_finetuned_score(data, model_path)

                # get event_weight
                if sample_to_use == "Data":
                    is_data = True
                else:
                    is_data = False

                event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)

                logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                # add nominal weight

                """"
                Make sure at least you're storing the GenWeight under "df[weight_{ch}]"
                """

                if is_data:  # for data (nominal is 1)
                    data[f"weight_{ch}"] = np.ones_like(data["fj_pt"])
                else:
                    data[f"weight_{ch}"] = data[f"weight_{ch}"] * event_weight

                hists.fill(
                    samples=sample_to_use,
                    var=data["rec_higgs_m"],
                    weight=data[f"weight_{ch}"],
                )

    logging.info(hists)

    return hists


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
