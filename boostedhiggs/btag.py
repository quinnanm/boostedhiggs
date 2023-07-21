import pathlib

import awkward as ak
import correctionlib
import numpy as np
from coffea import util as cutil
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.nanoaod import JetArray

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
}


def get_UL_year(year):
    if year == "2016":
        year = "2016postVFP"
    elif year == "2016APV":
        year = "2016preVFP"
    return f"{year}_UL"


def get_pog_json(obj, year):
    try:
        pog_json = pog_jsons[obj]
    except ValueError:
        print(f"No json for {obj}")
    year = get_UL_year(year)
    return f"{pog_correction_path}POG/{pog_json[0]}/{year}/{pog_json[1]}"
    # noqa: os.system(f"cp {pog_correction_path}POG/{pog_json[0]}/{year}/{pog_json[1]} boostedhiggs/data/POG_{pog_json[0]}_{year}_{pog_json[1]}")
    # fname = ""
    # with importlib.resources.path("boostedhiggs.data", f"POG_{pog_json[0]}_{year}_{pog_json[1]}") as filename:
    #     fname = str(filename)
    # print(fname)
    # return fname


def _btagSF(cset, jets, flavour, wp="M", algo="deepJet", syst="central"):
    j, nj = ak.flatten(jets), ak.num(jets)
    corrs = cset[f"{algo}_comb"] if flavour == "bc" else cset[f"{algo}_incl"]
    sf = corrs.evaluate(
        syst,
        wp,
        np.array(j.hadronFlavour),
        np.array(abs(j.eta)),
        np.array(j.pt),
    )
    return ak.unflatten(sf, nj)


def _btag_prod(eff, sf):
    num = ak.fill_none(ak.prod(1 - sf * eff, axis=-1), 1)
    den = ak.fill_none(ak.prod(1 - eff, axis=-1), 1)
    return num, den


def add_btag_weights(
    weights: Weights,
    year: str,
    candidatefj: JetArray,
    wp: str = "M",
    algo: str = "deepJet",
):
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("btagging", year))
    efflookup = cutil.load(package_path + f"/boostedhiggs/data/btageff_deepJet_M_{year}.coffea")

    lightJets = candidatefj[(candidatefj.hadronFlavour == 0)]
    bcJets = candidatefj[(candidatefj.hadronFlavour > 0)]

    lightEff = efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
    bcEff = efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

    lightSF = _btagSF(cset, lightJets, "light", wp, algo)
    bcSF = _btagSF(cset, bcJets, "bc", wp, algo)

    lightnum, lightden = _btag_prod(lightEff, lightSF)
    bcnum, bcden = _btag_prod(bcEff, bcSF)

    weight = np.nan_to_num((1 - lightnum * bcnum) / (1 - lightden * bcden), nan=1)
    weights.add("btagSF", weight)
