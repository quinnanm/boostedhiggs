"""
Loads the config from `config_make_stacked_hists.yaml`, and postprocesses
the condor output to make stacked histograms

Author: Farouk Mokhtar
"""
import argparse
import logging
import math
import os
import pickle as pkl
import warnings

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import utils
import yaml
from make_stacked_hists import make_events_dict

plt.rcParams.update({"font.size": 20})

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)
plt.style.use(hep.style.CMS)
pd.options.mode.chained_assignment = None


dominant_backgrounds = ["WJetsLNu", "TTbar", "QCD"]


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    PATH = args.outpath + args.tag
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    os.system(f"cp config_compute_SoverB.yaml {PATH}/")

    # load config from yaml
    with open("config_compute_SoverB.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    logging.info("##### PRESELECTION")
    for ch in config["presel"]:
        logging.info(f"{ch} CHANNEL")
        for sel, value in config["presel"][ch].items():
            logging.info(f"{sel}: {value}")
        logging.info("-----------------------------")

    if args.make_events_dict:
        events_dict = make_events_dict(
            years, channels, args.samples_dir, config["samples"], config["presel"], config["weights"], logging_=False
        )

        with open(f"{PATH}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{PATH}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.info("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    logging.info("##### SELECTIONS")
    for ch in config["sel"]:
        logging.info(f"{ch} CHANNEL")
        for sel, value in config["sel"][ch].items():
            logging.info(f"{sel}: {value}")
        logging.info("-----------------------------")

    num_sig, num_bkg = {}, {}
    num_bkg["Other"] = 0

    deno_sig, deno_bkg = 0, 0
    for year in years:
        samples = events_dict[year]["ele"].keys()

        for sample in samples:
            df_mu = events_dict["2017"]["mu"][sample]
            df_ele = events_dict["2017"]["ele"][sample]

            for sel, value in config["sel"]["mu"].items():
                df_mu = df_mu.query(value)

            for sel, value in config["sel"]["ele"].items():
                df_ele = df_ele.query(value)

            if sample in utils.signals:
                deno_sig += df_mu["event_weight"].sum() + df_ele["event_weight"].sum()
                num_sig[sample] = df_mu["event_weight"].sum() + df_ele["event_weight"].sum()
            else:
                deno_bkg += df_mu["event_weight"].sum() + df_ele["event_weight"].sum()
                if sample in dominant_backgrounds:
                    num_bkg[sample] = df_mu["event_weight"].sum() + df_ele["event_weight"].sum()
                else:
                    num_bkg["Other"] += df_mu["event_weight"].sum() + df_ele["event_weight"].sum()

    num_sig = dict(sorted(num_sig.items(), key=lambda item: item[1]))
    num_bkg = dict(sorted(num_bkg.items(), key=lambda item: item[1]))

    print(rf"s/b: {deno_sig/(deno_bkg):.3f}")
    print(rf"s/sqrt(b): {deno_sig/math.sqrt(deno_bkg):.2f}")
    print("------------------------")
    print(f"Signal: {deno_sig:.2f}")
    for sample in num_sig:
        print(f"- {sample}: {100*(num_sig[sample]/deno_sig):.0f}%")
    #     print(f"- {sample}: {(num_sig[sample]/deno_sig):.3f}%")

    print("------------------------")
    print(f"Background: {deno_bkg:.2f}")
    for sample in num_bkg:
        print(f"- {sample}: {100*(num_bkg[sample]/deno_bkg):.0f}%")
    #     print(f"- {sample}: {(num_bkg[sample]/deno_bkg):.3f}%")


if __name__ == "__main__":
    # e.g.
    # python compute_SoverB.py --years 2017 --channels mu,ele --make_events_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--samples_dir", dest="samples_dir", default="../eos/Jul21_", help="path to parquets", type=str)
    parser.add_argument("--outpath", dest="outpath", default="soverb/", help="path of the output", type=str)
    parser.add_argument("--tag", dest="tag", default="test", help="path of the output", type=str)
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="Make events dictionary", action="store_true")
    parser.add_argument("--plot_hists", dest="plot_hists", help="Plot histograms", action="store_true")

    args = parser.parse_args()

    main(args)
