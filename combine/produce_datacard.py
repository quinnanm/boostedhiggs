# flake8: noqa
from __future__ import division, print_function

import argparse
import os
import pickle
import sys

import numpy as np
import rhalphalib as rl
import ROOT
import scipy.stats

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    for year in years:
        for ch in channels:
            model_name = "model_{}_{}".format(year, ch)

            tmpdir = "{}/{}/{}.pkl".format("templates", args.tag, model_name)

            with open(tmpdir, "rb") as fout:
                model = pickle.load(fout)

            model.renderCombine(os.path.join(str("{}/{}".format("templates", args.tag)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python produce_datacard.py --years 2017 --channels mu --tag test

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--tag", dest="tag", default="test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
