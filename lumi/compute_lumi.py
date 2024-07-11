#!/usr/bin/python

import numpy as np
from coffea.lumi_tools import LumiData, LumiList

"""
This script computes the total luminosity using
    (1) a single lumi_set.pkl file saved using combine_lumi.py.
    (2) a lumi.csv produced using the GoldenJson with the commands below.

    # noqa for 2016: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt > lumi2016.csv
    # noqa for 2017: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt > lumi2017.csv
    # noqa for 2018: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt > lumi2018.csv
"""


def main():

    year = "2016APV"
    dir_ = f"/eos/uscms/store/user/fmokhtar/boostedhiggs/lumi_{year}/"

    import pickle

    with open(f"{dir_}/lumi_set.pkl", "rb") as f:
        lumi_set = pickle.load(f)

    print("------------------------------------")

    lumis = {}

    # combine the sets from the different datasets
    for i, dataset in enumerate(lumi_set.keys()):
        if "Muon" in dataset:
            ch = "mu"
        else:
            ch = "ele"

        print(f"Retrieving lumi_set of {dataset}")
        if ch not in lumis.keys():
            lumis[ch] = lumi_set[dataset]
        else:
            lumis[ch] = lumis[ch] | lumi_set[dataset]

    lumi_list = {}
    for ch in ["ele", "mu"]:
        # convert the set to a numpy 2d-array
        lumis[ch] = np.array(list(lumis[ch]))

        # make LumiList object
        lumi_list[ch] = LumiList(runs=lumis[ch][:, 0], lumis=lumis[ch][:, 1])

        # this csv was made using brilcalc and the GoldenJson2017... refer to
        # https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
        lumidata = LumiData(f"lumi{year}.csv")
        print(f"---> Lumi for {ch} channel = {lumidata.get_lumi(lumi_list[ch])}")
        print("------------------------------------")


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python compute_lumi.py

    main()
