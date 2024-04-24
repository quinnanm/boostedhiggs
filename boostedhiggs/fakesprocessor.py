import importlib.resources
import json
import logging
import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate

logger = logging.getLogger(__name__)

from boostedhiggs.corrections import (
    add_pileup_weight,
    add_pileupid_weights,
    btagWPs,
    get_jec_jets,
    met_factory,
)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


class FakesProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        apply_PR_sel=False,
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels

        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        if self._year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }

        self.jecs = {
            "JES": "JES_jes",
            "JER": "JER",
        }

        self._apply_PR_sel = apply_PR_sel

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = self._channels if channel == "all" else [channel]

        for ch in channels:
            if ch not in self._channels:
                logger.warning(f"Attempted to add selection to unexpected channel: {ch} not in %s" % (self._channels))
                continue

            # add selection
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            if self.isMC:
                weight = self.weights[ch].partial_weight(["genweight"])
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]
        self.isMC = hasattr(events, "genWeight")

        nevents = len(events)
        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents

        # add genweight before filling cutflow
        if self.isMC:
            for ch in self._channels:
                self.weights[ch].add("genweight", events.genWeight)

        ######################
        # Trigger
        ######################

        trigger, trigger_noiso, trigger_iso = {}, {}, {}
        for ch in self._channels:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            trigger_noiso[ch] = np.zeros(nevents, dtype="bool")
            trigger_iso[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    if "Iso" in t or "WPTight_Gsf" in t:
                        trigger_iso[ch] = trigger_iso[ch] | events.HLT[t]
                    else:
                        trigger_noiso[ch] = trigger_noiso[ch] | events.HLT[t]
                    trigger[ch] = trigger[ch] | events.HLT[t]

        ######################
        # METFLITERS
        ######################

        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        ######################
        # OBJECT DEFINITION
        ######################

        muons = ak.with_field(events.Muon, 0, "flavor")
        electrons = ak.with_field(events.Electron, 1, "flavor")

        # OBJECT: loose & tight muons
        loose_muons = (muons.pt > 30) & (np.abs(muons.eta) < 2.4) & (muons.looseId)

        tight_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & (((muons.pt < 55)) & (muons.pfRelIso04_all < 0.15) | (muons.pt >= 55))
        )

        n_loose_muons = ak.sum(loose_muons, axis=1)
        n_tight_muons = ak.sum(tight_muons, axis=1)

        # OBJECT: loose & tight electrons
        loose_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WPL)
        )

        tight_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WP90)
            # additional cuts
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
        )

        n_loose_electrons = ak.sum(loose_electrons, axis=1)
        n_tight_electrons = ak.sum(tight_electrons, axis=1)

        # OBJECT: loose leptons
        loose_leptons = ak.concatenate([muons[loose_muons], electrons[loose_electrons]], axis=1)
        loose_leptons = loose_leptons[ak.argsort(loose_leptons.pt, ascending=False)]  # sort by pt

        N_loose_lep = ak.num(loose_leptons)

        loose_lep1 = ak.firsts(loose_leptons[:, 0:1])  # pick highest pt (equivalent to ak.firsts())
        loose_lep2 = ak.firsts(loose_leptons[:, 1:2])  # pick second highest pt

        mll_loose = (loose_lep1 + loose_lep2).mass

        # OBJECT: tight leptons
        tight_leptons = ak.concatenate([muons[tight_muons], electrons[tight_electrons]], axis=1)
        tight_leptons = tight_leptons[ak.argsort(tight_leptons.pt, ascending=False)]  # sort by pt

        N_tight_lep = ak.num(tight_leptons)

        tight_lep1 = ak.firsts(tight_leptons[:, 0:1])  # pick highest pt (equivalent to ak.firsts())
        tight_lep2 = ak.firsts(tight_leptons[:, 1:2])  # pick second highest pt

        mll_tight = (tight_lep1 + tight_lep2).mass

        # OBJECT: AK4 jets
        jets, _ = get_jec_jets(events, events.Jet, self._year, not self.isMC, self.jecs, fatjets=False)
        met = met_factory.build(events.MET, jets, {}) if self.isMC else events.MET

        jet_selector = (
            (jets.pt > 30)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
        )
        goodjets = jets[jet_selector]

        # OBJECT: b-jets
        n_bjets_L = ak.sum(jets.btagDeepFlavB > btagWPs["deepJet"][self._year]["L"], axis=1)

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        good_fatjets, _ = get_jec_jets(events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True)
        NumFatjets = ak.num(good_fatjets)

        candidatelep_p4 = build_p4(loose_lep1)  # build p4 for candidate lepton

        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        variables = {
            "N_tight_lep": N_tight_lep,
            "N_loose_lep": N_loose_lep,
            # tight
            "tight_lep1_pt": tight_lep1.pt,
            "tight_lep1_eta": tight_lep1.eta,
            "tight_lep2_pt": tight_lep2.pt,
            "tight_lep2_eta": tight_lep2.eta,
            "mll_tight": mll_tight,
            # loose
            "loose_lep1_pt": loose_lep1.pt,
            "loose_lep1_eta": loose_lep1.eta,
            "loose_lep2_pt": loose_lep2.pt,
            "loose_lep2_eta": loose_lep2.eta,
            "mll_loose": mll_loose,
            # others
            "met_pt": met.pt,
            "NumFatjets": NumFatjets,
            "lep_fj_dr": lep_fj_dr,
            "fj_pt": candidatefj.pt,
            "fj_eta": candidatefj.eta,
            "fj_phi": candidatefj.phi,
        }

        for ch in self._channels:
            self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)
        self.add_selection(name="METFilters", sel=metfilters)
        self.add_selection(name="bveto", sel=(n_bjets_L == 0))

        if self._apply_PR_sel:  # apply PR selection
            # noqa: https://github.com/latinos/PlotsConfigurations/blob/5aca6c9e2f4ecee3ebcb60bf4d867578b07ae701/Configurations/monoHWW/SemiLep/Full2017_v7/PR/cuts.py

            self.add_selection(
                name="TwoLep", sel=(n_tight_muons > 0) & (n_loose_muons > 1) & (n_loose_electrons == 0), channel="mu"
            )
            self.add_selection(
                name="TwoLep", sel=(n_tight_electrons > 0) & (n_loose_electrons > 1) & (n_loose_muons == 0), channel="ele"
            )
            self.add_selection(name="oppositeCharge", sel=(loose_lep1.charge * loose_lep2.charge < 0))
            self.add_selection(name="Zpeak", sel=(mll_loose > 76) & (mll_loose < 106))

        else:  # apply FR selection
            self.add_selection(name="MET", sel=(met.pt < 30))
            self.add_selection(name="OneLep", sel=(n_loose_muons == 1) & (n_loose_electrons == 0), channel="mu")
            self.add_selection(name="OneLep", sel=(n_loose_muons == 0) & (n_loose_electrons == 1), channel="ele")

            self.add_selection(name="AtLeastOneFatJet", sel=(NumFatjets >= 1))
            self.add_selection(name="CandidateJetpT", sel=(candidatefj.pt > 250))
            self.add_selection(name="LepInJet", sel=(lep_fj_dr < 0.8))
            self.add_selection(name="JetLepOverlap", sel=(lep_fj_dr > 0.03))

        # hem-cleaning selection
        if self._year == "2018":
            hem_veto = ak.any(
                ((goodjets.eta > -3.2) & (goodjets.eta < -1.3) & (goodjets.phi > -1.57) & (goodjets.phi < -0.87)),
                -1,
            ) | ak.any(
                (
                    (electrons.pt > 30)
                    & (electrons.eta > -3.2)
                    & (electrons.eta < -1.3)
                    & (electrons.phi > -1.57)
                    & (electrons.phi < -0.87)
                ),
                -1,
            )

            hem_cleaning = (
                ((events.run >= 319077) & (not self.isMC))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (hem_veto)

            self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

        if self.isMC:
            for ch in self._channels:
                if self._year in ("2016", "2017"):
                    self.weights[ch].add(
                        "L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn
                    )
                add_pileup_weight(self.weights[ch], self._year, self._yearmod, nPU=ak.to_numpy(events.Pileup.nPU))

                add_pileupid_weights(self.weights[ch], self._year, self._yearmod, goodjets, events.GenJet, wp="L")

                # store the gen-weight
                variables[f"weight_{ch}"] = self.weights[ch].partial_weight(["genweight"])

                # store the final weight per ch
                variables[f"weight_{ch}"] = self.weights[ch].weight()

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {}
                for var, item in variables.items():
                    # pad all the variables that are not a cut with -1
                    # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    # fill out dictionary
                    out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year
                + self._yearmod: {
                    "sumgenweight": sumgenweight,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
