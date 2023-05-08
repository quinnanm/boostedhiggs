import importlib.resources
import json
import os
import pathlib
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate

from boostedhiggs.btag import btagWPs
from boostedhiggs.corrections import add_lepton_weight, add_pileup_weight, add_VJets_kFactors

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")

import logging

logger = logging.getLogger(__name__)


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


class ZllProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        apply_trigger=True,
        apply_selection=True,
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels

        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # apply selection?
        self.apply_selection = apply_selection

        # apply trigger in selection?
        self.apply_trigger = apply_trigger

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        # b-tagging corrector
        self._btagWPs = btagWPs["deepJet"][self._year + self._yearmod]
        # self._btagSF = BTagCorrector("M", "deepJet", year, yearmod)

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

        # do inference
        self.inference = inference
        # for tagger model and preprocessing dict
        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

        self.weights_per_ch = {
            "mu": [
                "trigger_iso_muon",
                "trigger_noniso_muon",
                "isolation_muon",
                "id_muon",
            ],
            "ele": ["reco_electron", "id_electron", "trigger_electron"],
        }
        if self._year in ("2016", "2017"):
            self.common_weights = ["genweight", "L1Prefiring", "pileup"]
        else:
            self.common_weights = ["genweight", "pileup"]

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
                # for MC: multiply cutflow by gen weight
                weight = self.weights.partial_weight(["genweight"])
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.selections = {}
        self.cutflows = {}
        for ch in self._channels:
            self.selections[ch] = PackedSelection()
            self.cutflows[ch] = {}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

        # trigger
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

        # metfilters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # define the two leptons
        muons = events.Muon
        electrons = events.Electron

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (electrons.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        lep1 = ak.firsts(goodleptons[:, 0:1])  # pick highest pt (equivalent to ak.firsts())
        lep2 = ak.firsts(goodleptons[:, 1:2])  # pick second highest pt

        mll = (lep1 + lep2).mass

        # lepton isolation
        lep1_reliso = lep1.pfRelIso04_all if hasattr(lep1, "pfRelIso04_all") else lep1.pfRelIso03_all
        lep2_reliso = lep2.pfRelIso04_all if hasattr(lep2, "pfRelIso04_all") else lep2.pfRelIso03_all

        variables = {
            "lep1_pt": lep1.pt,
            "lep1_mass": lep1.mass,
            "lep1_charge": lep1.charge,
            "lep2_pt": lep2.pt,
            "lep2_mass": lep2.mass,
            "lep2_charge": lep2.charge,
            "mll": mll,
        }

        """
        HEM issue: Hadronic calorimeter Endcaps Minus (HEM) issue.
        The endcaps of the hadron calorimeter failed to cover the phase space at -3 < eta < -1.3 and -1.57 < phi < -0.87
        during the 2018 data C and D.
        The transverse momentum of the jets in this region is typically under-measured, this results in over-measured MET.
        It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = (
                ((events.run >= 319077) & ~self.isMC)  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (
                ak.any(
                    (
                        (events.Jet.pt > 30.0)
                        & (events.Jet.eta > -3.2)
                        & (events.Jet.eta < -1.3)
                        & (events.Jet.phi > -1.57)
                        & (events.Jet.phi < -0.87)
                    ),
                    -1,
                )
                | ((events.MET.phi > -1.62) & (events.MET.pt < 470.0) & (events.MET.phi < -0.62))
            )

            variables["hem_cleaning"] = hem_cleaning

        if self.apply_trigger:
            for ch in self._channels:
                self.add_selection(name="trigger", sel=trigger[ch], channel=ch)
        self.add_selection(name="metfilters", sel=metfilters)

        # lepton kinematic selection
        self.add_selection(name="leptonKin", sel=(lep1.pt > 30) & (lep2.pt > 30), channel="mu")
        self.add_selection(name="leptonKin", sel=(lep1.pt > 40) & (lep2.pt > 40), channel="ele")

        # dilepton selection
        self.add_selection(name="twoLepton", sel=(n_good_muons >= 2), channel="mu")
        self.add_selection(name="twoLepton", sel=(n_good_electrons >= 2), channel="ele")
        self.add_selection(name="opposite_charge", sel=(lep1.charge * lep2.charge < 0))

        # lepton isolation selection
        self.add_selection(
            name="lep_isolation",
            sel=(
                ((lep1.pt < 120) & (lep1_reliso < 0.15)) | (lep1.pt >= 120),
                ((lep2.pt < 120) & (lep2_reliso < 0.15)) | (lep2.pt >= 120),
            ),
            channel="ele",
        )
        self.add_selection(
            name="lep_isolation",
            sel=(
                ((lep1.pt < 55) & (lep1_reliso < 0.15)) | (lep1.pt >= 55),
                ((lep2.pt < 55) & (lep2_reliso < 0.15)) | (lep2.pt >= 55),
            ),
            channel="mu",
        )

        # gen-level matching
        if self.isMC:
            genVars = {}
            variables = {**variables, **genVars}

        """
        Weights
        ------
        - Gen weight
        - Pileup weight
        - L1 prefiring weight for 2016/2017
        - B-tagging efficiency weights (ToDo)
        - Electron trigger scale factors
        - Muon trigger scale factors
        - Electron ID scale factors and Reco scale factors
        - Muon ID scale factors
        - Muon Isolation scale factors
        - Electron Isolation scale factors (ToDo)
        - Mini-isolation scale factor (ToDo)
        - Jet Mass Scale (JMS) scale factor (ToDo)
        - Jet Mass Resolution (JMR) scale factor (ToDo)
        - NLO EWK scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq)
        - ~NNLO QCD scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq)
        - LHE scale weights for signal (ToDo)
        - LHE pdf weights for signal (ToDo)
        - PSweights for signal (ToDo)
        - ParticleNet tagger efficienc (ToDo) y

        Up and Down Variations (systematics included as a new variable)
        ----
        - Pileup weight Up/Down
        - L1 prefiring weight Up/Down
        - B-tagging efficiency Up/Down (ToDo)
        - Electron Trigger Up/Down (ToDo)
        - Muon Trigger Up/Down
        - Electron ID Up/Down
        - Electron Isolation Up/Down
        - Muon ID Up/Down
        - Muon Isolation Up/Down
        - JMS Up/Down
        - JMR Up/Down
        - LHE scale variations for signal
        - LHE pdf weights for signal
        - PSweights variations for signal
        - ParticleNet tagger Up/Down
        Up and Down Variations (systematics included as a new output file)
        ----
        - Jet Energy Scale (JES)
        - Jet Energy Resolution (JER)
        - MET unclustered up/down
        """
        if self.isMC:
            self.weights.add("genweight", events.genWeight)

            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring",
                    events.L1PreFiringWeight.Nom,
                    events.L1PreFiringWeight.Up,
                    events.L1PreFiringWeight.Dn,
                )

            add_pileup_weight(
                self.weights,
                self._year,
                self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )

            add_lepton_weight(self.weights, lep1, self._year + self._yearmod, "muon")
            add_lepton_weight(self.weights, lep2, self._year + self._yearmod, "muon")

            add_lepton_weight(self.weights, lep1, self._year + self._yearmod, "electron")
            add_lepton_weight(self.weights, lep2, self._year + self._yearmod, "electron")

            add_VJets_kFactors(self.weights, events.GenPart, dataset)

            # store the final common weight
            variables["weight"] = self.weights.partial_weight(self.common_weights)

            for key in self.weights._weights.keys():
                # ignore btagSFlight/bc for now
                if "btagSFlight" in key or "btagSFbc" in key:
                    continue

                # store the individual weights (ONLY for now until we debug)
                variables[f"weight_{key}"] = self.weights.partial_weight([key])

            # NOTE: to add variations:
            # for var in self.weights.variations:
            #     variables["common"][f"weight_{key}"] = self.weights.weight(key)

        # initialize pandas dataframe
        output = {}

        for ch in self._channels:
            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            if self.apply_selection:
                selection_ch = self.selections[ch].all(*self.selections[ch].names)
                # only fill output for that channel if the selections yield any events
                if np.sum(selection_ch) <= 0:
                    fill_output = False
            else:
                selection_ch = np.ones(nevents, dtype="bool")

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
