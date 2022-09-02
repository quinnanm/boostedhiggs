from coffea.analysis_tools import Weights, PackedSelection
from coffea.nanoevents.methods import candidate, vector
from coffea.processor import ProcessorABC, column_accumulator
import pandas as pd
import numpy as np
import warnings
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from hist.intervals import clopper_pearson_interval

# we suppress ROOT warnings where our input ROOT tree has duplicate branches - these are handled correctly.
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def getParticles(genparticles, lowid=22, highid=25, flags=["fromHardProcess", "isLastCopy"]):
    """
    returns the particle objects that satisfy a low id,
    high id condition and have certain flags
    """
    absid = abs(genparticles.pdgId)
    return genparticles[((absid >= lowid) & (absid <= highid)) & genparticles.hasFlags(flags)]


def simple_match_HWW(genparticles, candidatefj):
    """
    return the number of matched objects (hWW*),daughters,
    and gen flavor (enuqq, munuqq, taunuqq)
    """
    higgs = getParticles(genparticles, 25)  # genparticles is the full set... this function selects Higgs particles
    # W~24 so we get H->WW (limitation: only picking one W and assumes the other will be there)
    is_hWW = ak.all(abs(higgs.children.pdgId) == 24, axis=2)

    higgs = higgs[is_hWW]

    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)  # choose higgs closest to fj

    return matchedH


class TriggerEfficienciesProcessor(ProcessorABC):
    """Accumulates yields from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self, year=2017):
        super(TriggerEfficienciesProcessor, self).__init__()
        self._year = year
        self._trigger_dict = {
            2017: {
                "ele35": [
                    "Ele35_WPTight_Gsf",
                ],
                "ele115": ["Ele115_CaloIdVT_GsfTrkIdT"],
                "Photon200": ["Photon200"],
                "Mu50": [
                    "Mu50",
                ],
                "IsoMu27": ["IsoMu27"],
                "OldMu100": ["OldMu100"],
                "TkMu100": ["TkMu100"],
            }
        }[self._year]
        self._triggers = {
            "ele": ["ele35", "ele115", "Photon200"],
            "mu": ["Mu50", "IsoMu27", "OldMu100", "TkMu100"],
        }

        self._channels = ["ele", "mu"]

    def pad_val(self, arr: ak.Array, target: int, value: float, axis: int = 0, to_numpy: bool = True):
        """pads awkward array up to `target` index along axis `axis` with value `value`, optionally converts to numpy array"""
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=True), value)
        return ret.to_numpy() if to_numpy else ret

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger histograms from input NanoAOD events"""
        dataset = events.metadata["dataset"]
        n_events = len(events)
        isRealData = not hasattr(events, "genWeight")

        def pad_val_nevents(arr: ak.Array):
            """pad values with the length equal to the number of events"""
            return self.pad_val(arr, n_events, -1)

        # skimmed events for different channels
        out = {}
        for channel in self._channels:
            out[channel] = {}

        """ Save OR of triggers as booleans """
        for channel in self._channels:
            HLT_triggers = {}
            for t in self._triggers[channel]:
                HLT_triggers["HLT_" + t] = np.any(
                    np.array([events.HLT[trigger] for trigger in self._trigger_dict[t] if trigger in events.HLT.fields]),
                    axis=0,
                )
            out[channel] = {**out[channel], **HLT_triggers}

        """ Baseline selection """
        goodmuon = (events.Muon.pt > 25) & (abs(events.Muon.eta) < 2.4) & events.Muon.mediumId
        nmuons = ak.sum(goodmuon, axis=1)
        goodelectron = (events.Electron.pt > 25) & (abs(events.Electron.eta) < 2.5) & (events.Electron.mvaFall17V2noIso_WP90)
        nelectrons = ak.sum(goodelectron, axis=1)

        # taus (will need to refine to avoid overlap with htt)
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        # leading lepton
        goodleptons = ak.concatenate([events.Muon[goodmuon], events.Electron[goodelectron]], axis=1)
        candidatelep = ak.firsts(goodleptons[ak.argsort(goodleptons.pt)])

        # fatjet closest to MET
        fatjets = events.FatJet
        candidatefj = fatjets[(fatjets.pt > 200) & (abs(fatjets.eta) < 2.4)]
        met = events.MET
        dphi_met_fj = abs(candidatefj.delta_phi(met))
        candidatefj = ak.firsts(candidatefj[ak.argmin(dphi_met_fj, axis=1, keepdims=True)])
        dr_lep_fj = candidatefj.delta_r(candidatelep)

        # jets
        jets = events.Jet
        candidatejet = jets[(jets.pt > 30) & (abs(jets.eta) < 2.5) & jets.isTight]

        # define isolation
        mu_iso = ak.where(candidatelep.pt >= 55.0, candidatelep.miniPFRelIso_all, candidatelep.pfRelIso03_all)
        ele_iso = ak.where(candidatelep.pt >= 120.0, candidatelep.pfRelIso03_all, candidatelep.pfRelIso03_all)

        # define selections for different channels
        for channel in self._channels:
            selection = PackedSelection()
            selection.add("fjkin", candidatefj.pt > 200)
            if channel == "mu":
                selection.add("onemuon", (nmuons == 1) & (nelectrons == 0) & (n_loose_taus_mu == 0))
                selection.add("muonkin", (candidatelep.pt > 27.0) & abs(candidatelep.eta < 2.4))
            elif channel == "ele":
                selection.add("oneelectron", (nelectrons == 1) & (nmuons == 0) & (n_loose_taus_ele == 0))
                selection.add("electronkin", (candidatelep.pt > 30.0) & abs(candidatelep.eta < 2.4))

            """ Define other variables to save """
            out[channel]["fj_pt"] = pad_val_nevents(candidatefj.pt)
            out[channel]["fj_msoftdrop"] = pad_val_nevents(candidatefj.msoftdrop)
            out[channel]["lep_pt"] = pad_val_nevents(candidatelep.pt)

            if "HToWW" in dataset:
                matchedH = simple_match_HWW(events.GenPart, candidatefj)
                matchedH_pt = ak.firsts(matchedH.pt)
            else:
                matchedH_pt = ak.zeros_like(candidatefj.pt)
            out[channel]["higgspt"] = pad_val_nevents(matchedH_pt)

            # use column accumulators
            out[channel] = {
                key: column_accumulator(value[selection.all(*selection.names)]) for (key, value) in out[channel].items()
            }

        return {self._year: {dataset: {"nevents": n_events, "skimmed_events": out}}}

    def postprocess(self, accumulator):
        for year, datasets in accumulator.items():
            for dataset, output in datasets.items():
                for channel in output["skimmed_events"].keys():
                    output["skimmed_events"][channel] = {
                        key: value.value for (key, value) in output["skimmed_events"][channel].items()
                    }

        return accumulator

        # now save pandas dataframes
        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")

        # return dictionary with cutflows
        return {dataset: {"mc": isMC, self._year: {"sumgenweight": sumgenweight, "cutflows": self.cutflows}}}

    def postprocess(self, accumulator):
        return accumulator
