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
import utils
import yaml

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def make_events_dict(
    years,
    channels,
    samples_dir,
    samples,
    presel,
    weights,
    columns="all",
    add_tagger_score=False,
):
    """
    Postprocess the parquets by applying preselections, saving an event_weight column,
    saving the tagger score, and only keeping relevant columns in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets (note: the year will be appended to the string)
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)
        weights [dict]: weights to include in the event_weight per ch (e.g. `weights = {"mu": {"weight_genweight": 1}})
        columns [list]: relevant columns of the parquets to keep (default="all" which keeps all columns)
        add_tagger_score [Bool]: adds a column which is the tagger score (must be True if a tagger cut is in the presel)

    Returns
        a dict() object events_dict[year][channel][samples] that contains big dataframes of procesed events

    """

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            events_dict[year][ch] = {}

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

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
                # # TODO: draw distribution of number of primary vertices before and after applying this weight
                # if not is_data:
                #     strange_events = data["weight_pileup"] > 6
                #     if len(strange_events) > 0:
                #         data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

                # get event_weight
                if not is_data:
                    event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                    data["lol"] = event_weight
                    if "Jul21_" in samples_dir:
                        print("---> Using already stored event weight")
                        event_weight = data[f"weight_{ch}"]
                    else:
                        print("---> Accumulating event weights.")
                        for w in weights[ch]:
                            if w not in data.keys():
                                print(f"{w} weight is not stored in parquet")
                                continue
                            if weights[ch][w] == 1:
                                print(f"Applying {w} weight")
                                # event_weight *= data[w]
                                event_weight = data[w]

                        print("---> Done with accumulating event weights.")
                else:
                    event_weight = np.ones_like(data["fj_pt"])
                    data["lol"] = event_weight

                data["event_weight"] = event_weight

                # add tagger scores
                if add_tagger_score:
                    if "Apr12_presel" in samples_dir:
                        data["inclusive_score"] = utils.disc_score(data, utils.new_sig, utils.inclusive_bkg)
                    else:
                        data["inclusive_score"] = data["fj_ParT_inclusive_score"]

                # apply selection
                print("---> Applying preselection.")
                for selection in presel[ch]:
                    print(f"applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])
                print("---> Done with preselection.")

                print(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
                print(f"tot event weight {data['event_weight'].sum()} \n")

                if columns == "all":
                    # fill the big dataframe
                    if sample_to_use not in events_dict[year][ch]:
                        events_dict[year][ch][sample_to_use] = data
                    else:
                        events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data])
                else:
                    # specify columns to keep
                    cols = columns + ["event_weight"]
                    if add_tagger_score:
                        cols += ["inclusive_score"]

                    # fill the big dataframe
                    if sample_to_use not in events_dict[year][ch]:
                        events_dict[year][ch][sample_to_use] = data[cols]
                    else:
                        events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data[cols]])

    return events_dict


def make_hists_from_events_dict(events_dict, samples_to_plot, vars_to_plot):
    """
    Takes an `events_dict` object that was processed using `make_events_dict` and starts filling histograms.

    """

    hists = {}
    for var in vars_to_plot:
        hists[var] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            utils.axis_dict[var],
        )

        for sample in samples_to_plot:
            for year in events_dict:
                for ch in events_dict[year]:
                    df = events_dict[year][ch][sample]

                    hists[var].fill(samples=sample, var=df[var], weight=df["event_weight"])

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp make_hists_config.yaml {args.outpath}/")

    # load config from yaml
    with open("make_hists_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.make_events_dict:
        events_dict = make_events_dict(
            years,
            channels,
            args.samples_dir,
            config["samples"],
            config["presel"],
            config["weights"],
            columns="all",
            add_tagger_score=args.add_score,
        )
        with open(f"{args.outpath}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{args.outpath}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            print("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    if args.plot_hists:
        hists = make_hists_from_events_dict(events_dict, config["samples_to_plot"], config["vars_to_plot"])

        utils.plot_hists(
            years,
            channels,
            hists,
            config["vars_to_plot"],
            add_data=False,
            logy=False,
            add_soverb=True,
            only_sig=False,
            mult=100,
            outpath=f"{args.outpath}/hists/",
        )


if __name__ == "__main__":
    # e.g.
    # python make_hists.py --years 2017 --channels ele,mu --make_events_dict --plot_hists

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="ele", help="channels separated by commas")
    parser.add_argument(
        "--samples_dir", dest="samples_dir", default="../eos/Apr12_presel_", help="path to parquets", type=str
    )
    parser.add_argument(
        "--outpath", dest="outpath", default="/Users/fmokhtar/Desktop/hww/test", help="path of the output", type=str
    )
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="Make events dictionary", action="store_true")
    parser.add_argument("--plot_hists", dest="plot_hists", help="Plot histograms", action="store_true")
    parser.add_argument("--add_score", dest="add_score", help="Add column of inclusive tagger score", action="store_true")

    args = parser.parse_args()

    main(args)
