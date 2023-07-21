#!/usr/bin/python

import argparse
import glob
import json
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import uproot
import utils
import yaml

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def make_templates(years, channels, samples_dir, samples, presel, weights, regions_selections):
    """
    Postprocess the parquets by applying preselections, saving an event_weight column, and
    fills histograms/templates for different regions.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets (note: the year will be appended to the string)
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)
        weights [dict]: weights to include in the event_weight per ch (e.g. `weights = {"mu": {"weight_genweight": 1}})
        regions_selections [dict]: (e.g. `{"signal_region": ( (inclusive_score>0.99) & (n_bjets_M < 2) )}`)

    Returns
        a dict() object hists[region] that contains histograms

    """

    hists = {}
    for region in regions_selections:
        hists[region] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            hist2.axis.Regular(20, 200, 600, name="fj_pt", label=r"Jet $p_T$ [GeV]", overflow=True),
            hist2.axis.Regular(35, 0, 480, name="rec_higgs_m", label=r"Higgs reconstructed mass [GeV]", overflow=True),
        )

    for year in years:
        print(f"Processing year {year}")
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
                    print(f"ATTENTION: {sample} will be skipped")
                    continue

                is_data = False
                if sample_to_use == "Data":
                    is_data = True

                print(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir + year}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    print(f"No parquet file for {sample}")
                    continue

                data = pd.read_parquet(parquet_files)
                if len(data) == 0:
                    continue

                # replace the weight_pileup of the strange events with the mean weight_pileup of all the other events
                if not is_data:
                    strange_events = data["weight_pileup"] > 6
                    if len(strange_events) > 0:
                        data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

                # apply selection
                for selection in presel[ch]:
                    data = data.query(presel[ch][selection])

                # get event_weight
                if not is_data:
                    event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                    for w in weights[ch]:
                        if w not in data.keys():
                            continue
                        if weights[ch][w] == 1:
                            event_weight *= data[w]
                else:
                    event_weight = np.ones_like(data["fj_pt"])

                data["event_weight"] = event_weight

                # add tagger scores
                if "Apr12_presel" in samples_dir:
                    data["inclusive_score"] = utils.disc_score(data, utils.new_sig, utils.inclusive_bkg)
                else:
                    data["inclusive_score"] = data["fj_ParT_inclusive_score"]

                for region in regions_selections:
                    data1 = data.copy()  # get fresh copy of the data to apply selections on
                    data1 = data1.query(regions_selections[region])

                    hists[region].fill(
                        samples=sample_to_use,
                        fj_pt=data1["fj_pt"],
                        rec_higgs_m=data1["rec_higgs_m"],
                        weight=data1["event_weight"],
                    )
        print("-------------------------------------------------")
    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp make_combine_templates_config.yaml {args.outpath}/")

    # load config from yaml
    with open("make_combine_templates_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    hists = make_templates(
        years,
        channels,
        args.samples_dir,
        config["samples"],
        config["presel"],
        config["weights"],
        config["regions_selections"],
    )

    with open(f"{args.outpath}/hists_templates.pkl", "wb") as fp:
        pkl.dump(hists, fp)

    # dump the templates of each region in a rootfile
    if not os.path.exists(f"{args.outpath}/hists_templates"):
        os.makedirs(f"{args.outpath}/hists_templates")

    for region in hists.keys():
        file = uproot.recreate(f"{args.outpath}/hists_templates/{region}.root")

        for sample in hists[region].axes["samples"]:
            if sample == "Data":
                file[f"{region}/data_obs"] = hists[region][{"fj_pt": sum, "samples": sample}]
                continue
            file[f"{region}/{sample}"] = hists[region][{"fj_pt": sum, "samples": sample}]


if __name__ == "__main__":
    # e.g.
    # python make_combine_templates.py --years 2017 --channels ele,mu

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="ele", help="channels separated by commas")
    parser.add_argument(
        "--samples_dir", dest="samples_dir", default="../eos/Apr12_presel_", type=str, help="path to parquets"
    )
    parser.add_argument(
        "--outpath", dest="outpath", default="/Users/fmokhtar/Desktop/hww/test", type=str, help="path of the output"
    )
    args = parser.parse_args()

    main(args)
