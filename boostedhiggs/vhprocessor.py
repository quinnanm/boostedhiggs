from collections import defaultdict
import pickle as pkl
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import shutil
import pathlib
from typing import List, Optional
import pyarrow.parquet as pq

import importlib.resources

from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

from boostedhiggs.utils import match_H, match_V, match_Top
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    add_VJets_kFactors,
    add_jetTriggerSF,
    add_lepton_weight,
    add_pileup_weight,
)
from boostedhiggs.btag import btagWPs, BTagCorrector

from .run_tagger_inference import runInferenceTriton

import warnings

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


def zleptons(good_leptons):
    ngood_leptons = ak.num(good_leptons, axis=1)

    min_three_leptons = ak.mask(good_leptons, (ngood_leptons >= 3)[:, None])

    lepton_pairs = ak.argcombinations(min_three_leptons, 2, fields=["first", "second"])
    lepton_pairs = ak.fill_none(lepton_pairs, [], axis=0)

    OSSF_pairs = lepton_pairs[
        (
            min_three_leptons[lepton_pairs["first"]].charge
            != min_three_leptons[lepton_pairs["second"]].charge
        )
        & (
            min_three_leptons[lepton_pairs["first"]].flavor
            == min_three_leptons[lepton_pairs["second"]].flavor
        )
    ]

    closest_pairs = OSSF_pairs[
        ak.local_index(OSSF_pairs)
        == ak.argmin(
            np.abs(
                (
                    min_three_leptons[OSSF_pairs["first"]]
                    + min_three_leptons[OSSF_pairs["second"]]
                ).mass
                - 91.2
            ),
            axis=1,
        )
    ]
    closest_pairs = ak.fill_none(closest_pairs, [], axis=0)

    # invariant Z mass
    ZLeptonMass = (
        min_three_leptons[closest_pairs.first] + min_three_leptons[closest_pairs.second]
    ).mass
    desired_length = np.max(ak.num(ZLeptonMass))

    ZLeptonMass = ak.ravel(
        ak.to_numpy(ak.fill_none(ak.pad_none(ZLeptonMass, desired_length), 0))
    )

    remainingLeptons = min_three_leptons[
        (ak.local_index(min_three_leptons) != ak.any(closest_pairs.first, axis=1))
        & (ak.local_index(min_three_leptons) != ak.any(closest_pairs.second, axis=1))
    ]

    return ZLeptonMass, remainingLeptons


def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(
            ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None
        )
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret


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


class vhProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year,
        yearmod="",
        output_location="./outfiles/",
        inference=False,
        apply_trigger=True,
    ):

        self._year = year
        self._yearmod = yearmod
        self._output_location = output_location

        # dictionary of trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # list of triggers
        if self._year == "2018":
            self._trigger_list = ["ele", "mu", "MuonEG"]
        else:
            self._trigger_list = ["ele", "mu", "DoubleMuon", "MuonEG", "DoubleEG"]

        # apply trigger in selection?
        self.apply_trigger = apply_trigger

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        # b-tagging corrector
        self._btagWPs = btagWPs["deepJet"][year + yearmod]
        # self._btagSF = BTagCorrector("M", "deepJet", year, yearmod)

        # datasets and match them to a main set of triggers
        if year == "2018":
            self.dataset = {
                "SingleMuon": ["mu"],
                "DoubleMuon": ["DoubleMuon"],
                "EGamma": ["ele"],  # 2018 doesn't have single electron
                "MuonEG": ["MuonEG"],
            }
        else:
            self.dataset = {
                "SingleElectron": ["ele"],
                "SingleMuon": ["mu"],
                "DoubleMuon": ["DoubleMuon"],
                "DoubleEG": ["DoubleEG"],
                "MuonEG": ["DoubleEG"],
            }

        # do inference
        self.inference = inference

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        # do trigger efficiency study
        self.trigger_eff_study = False

        self.common_weights = ["genweight", "L1Prefiring", "pileup"]

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(
                    table, self._output_location + "/parquet/" + fname + ".parquet"
                )

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        self.selections.add(name, sel)
        selection = self.selections.all(*self.selections.names)
        if self.isMC:
            weight = self.weights.partial_weight(self.common_weights)
            self.cutflows[name] = float(weight[selection].sum())
        else:
            self.cutflows[name] = np.sum(selection)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata["dataset"]
        nevents = len(events)

        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.selections = {}
        self.cutflows = {}
        self.selections = PackedSelection()

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

        # dictionary of triggers
        trigger = {}
        for trig in self._trigger_list:
            trigger[trig] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[trig]:
                if t in events.HLT.fields:
                    trigger[trig] = trigger[trig] | events.HLT[t]

        # build a trigger logic per dataset
        trigger_logic = {}
        for dset, dtriggers in self.dataset.items():
            trigger_logic[dset] = np.ones(nevents, dtype="bool")
            for trig in self._trigger_list:
                if trig in dtriggers:
                    trigger_logic[dset] = trigger_logic[dset] | trigger[trig]
                else:
                    # if the trigger is not in the list of trigger per dataset
                    # then require that the events do not pass those other triggers
                    # to avoid overlaps
                    trigger_logic[dset] = trigger_logic[dset] & ~trigger[trig]

        if dataset in self.dataset.keys():
            trigger_decision = trigger_logic[dataset]
        else:
            trigger_decision = np.ones(nevents, dtype="bool")

        # metfilters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # add flavor tag to muons and electrons
        muons = ak.with_field(events.Muon, 0, "flavor")
        electrons = ak.with_field(events.Electron, 1, "flavor")
        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
        )

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (electrons.mvaFall17V2noIso_WP90)
        )

        good_leptons = ak.concatenate(
            [muons[good_muons], electrons[good_electrons]], axis=1
        )
        good_leptons = good_leptons[
            ak.argsort(good_leptons.pt, ascending=False)
        ]  # sort by pt

        ngood_leptons = ak.num(good_leptons, axis=1)

        # invariant Z mass
        ZLeptonMass, remainingLeptons = zleptons(good_leptons)

        candidatelep = ak.firsts(
            remainingLeptons
        )  # pick remaining lepton with highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        lep_reliso = (
            candidatelep.pfRelIso04_all
            if hasattr(candidatelep, "pfRelIso04_all")
            else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton
        mu_mvaId = (
            candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)
        )  # MVA-ID for candidate lepton

        # jets
        goodjets = events.Jet[
            (events.Jet.pt > 30)
            & (abs(events.Jet.eta) < 5.0)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        ]
        # reject EE noisy jets for 2017
        if self._year == "2017":
            goodjets = goodjets[
                (goodjets.pt > 50)
                | (abs(goodjets.eta) < 2.65)
                | (abs(goodjets.eta) > 3.139)
            ]
        ht = ak.sum(goodjets.pt, axis=1)

        # fatjets
        fatjets = events.FatJet
        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[
            ak.argsort(good_fatjets.pt, ascending=False)
        ]  # sort them by pt

        # for leptonic channel: first clean jets and leptons by removing overlap, then pick candidate_fj closest to the lepton
        lep_in_fj_overlap_bool = good_fatjets.delta_r(candidatelep_p4) > 0.1
        good_fatjets = good_fatjets[lep_in_fj_overlap_bool]
        fj_idx_lep = ak.argmin(
            good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True
        )
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # MET
        met = events.MET
        mt_lep_met = np.sqrt(
            2.0
            * candidatelep_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # delta phi MET and higgs candidate
        met_fjlep_dphi = candidatefj.delta_phi(met)

        # lepton and fatjet mass
        lep_fj_m = (candidatefj - candidatelep_p4).mass  # mass of fatjet without lepton

        # b-jets
        # in event, pick highest b score in opposite direction from signal (we will make cut here to avoid tt background events producing bjets)
        dphi_jet_lepfj = abs(goodjets.delta_phi(candidatefj))
        bjets_away_lepfj = goodjets[dphi_jet_lepfj > np.pi / 2]

        # deltaR
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        # VBF variables
        ak4_outside_ak8 = goodjets[goodjets.delta_r(candidatefj) > 0.8]
        n_jets_outside_ak8 = ak.sum(goodjets.delta_r(candidatefj) > 0.8, axis=1)
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass

        """
        HEM issue: Hadronic calorimeter Endcaps Minus (HEM) issue.
        The endcaps of the hadron calorimeter failed to cover the phase space at -3 < eta < -1.3 and -1.57 < phi < -0.87 during the 2018 data C and D.
        The transverse momentum of the jets in this region is typically under-measured, this results in over-measured MET. It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = events.run >= 319077 & ak.any(
                (
                    (events.Jet.pt > 30.0)
                    & (events.Jet.eta > -3.2)
                    & (events.Jet.eta < -1.3)
                    & (events.Jet.phi > -1.57)
                    & (events.Jet.phi < -0.87)
                ),
                -1,
            ) | ((met.phi > -1.62) & (met.pt < 470.0) & (met.phi < -0.62))

        # output tuple variables
        variables = {
            "Zmass": ZLeptonMass,
            "lepton_pT": candidatelep.pt,
            "fj_pt": candidatefj.pt,
            "lep_fj_dr": lep_fj_dr,
            "met": met.pt,
            "ht": ht,
        }

        # gen matching
        if self.isMC:
            if ("HToWW" in dataset) or ("HWW" in dataset) or ("ttHToNonbb" in dataset):
                genVars, signal_mask = match_H(events.GenPart, candidatefj)
                self.add_selection(name="signal", sel=signal_mask)
            elif "HToTauTau" in dataset:
                genVars, signal_mask = match_H(
                    events.GenPart, candidatefj, dau_pdgid=15
                )
                self.add_selection(name="signal", sel=signal_mask)
            elif ("WJets" in dataset) or ("ZJets" in dataset) or ("DYJets" in dataset):
                genVars = match_V(events.GenPart, candidatefj)
            elif "TT" in dataset:
                genVars = match_Top(events.GenPart, candidatefj)
            else:
                genVars = {}
            variables = {**variables, **genVars}

        # let's save the hem veto as a cut for now
        if self._year == "2018":
            variables["hem_cleaning"] = hem_cleaning

        """
        Weights
        ------
        - Gen weight (DONE)
        - Pileup weight (DONE)
        - L1 prefiring weight for 2016/2017 (DONE)
        - B-tagging efficiency weights 
        - Electron trigger scale factors 
        - Muon trigger scale factors 
        - Electron ID scale factors and Reco scale factors
        - Muon ID scale factors
        - Muon Isolation scale factors
        - Electron Isolation scale factors
        - Jet Mass Scale (JMS) scale factor
        - Jet Mass Resolution (JMR) scale factor
        - NLO EWK scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq)
        - ~NNLO QCD scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq)
        - LHE scale weights for signal
        - LHE pdf weights for signal
        - PSweights for signal
        - ParticleNet tagger efficiency
        Up and Down Variations (systematics included as a new variable)
        ----
        - Pileup weight Up/Down
        - L1 prefiring weight Up/Down
        - B-tagging efficiency Up/Down
        - Electron Trigger Up/Down
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

            variables["weight"] = self.weights.partial_weight(self.common_weights)

        """
        Selection and cutflows.
        """
        self.add_selection("all", np.ones(nevents, dtype="bool"))
        if self.trigger_eff_study:
            self.add_selection(
                name="OneOrMoreLeptons",
                sel=(ngood_leptons >= 1),
            )
        else:
            if self.apply_trigger:
                self.add_selection("trigger", trigger_decision)
            self.add_selection("metfilters", metfilters)
            self.add_selection(name="ht", sel=(ht > 200))
            # self.add_selection(
            #     name="antibjettag",
            #    sel=(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1) < self._btagWPs["M"]),
            # )
            self.add_selection(
                name="threeOrMoreLeptons",
                sel=(ngood_leptons >= 3),
            )
            self.add_selection(name="leptonKin", sel=(candidatelep.pt > 30))
            self.add_selection(name="fatjetKin", sel=candidatefj.pt > 200)
            self.add_selection(name="leptonInJet", sel=(lep_fj_dr < 0.8))

        # initialize pandas dataframe
        output = {}

        # fill output dictionary
        fill_output = True
        # for data, only fill output for the dataset needed
        if not self.isMC and dataset not in self.dataset.keys():
            fill_output = False

        selection = self.selections.all(*self.selections.names)
        # only fill output if the selections yield any events
        if np.sum(selection) <= 0:
            fill_output = False

        if fill_output:
            out = {}
            for var, item in variables.items():
                # pad all the variables that are not a cut with -1
                pad_item = (
                    item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                )
                # fill out dictionary
                out[var] = item

            # fill the output dictionary after selections
            # the line below with output is the one giving me issues
            output = {key: value[selection] for (key, value) in out.items()}

            # fill inference
            if self.inference:
                # print("pre-inference")
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path, events[selection], fj_idx_lep[selection]
                )
                #  print("post-inference")
                # print(pnet_vars)
                output = {
                    **output,
                    **{key: value for (key, value) in pnet_vars.items()},
                }
        else:
            output = {}

        # convert arrays to pandas
        if not isinstance(output, pd.DataFrame):
            output = self.ak_to_pandas(output)  # ak.to_dataframe
            # print("DF", output)

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        # for ch in self._channels:  # creating directories for each channel
        if not os.path.exists(self._output_location):
            os.makedirs(self._output_location)
        if not os.path.exists(self._output_location + "/parquet"):
            os.makedirs(self._output_location + "/parquet")

        self.save_dfs_parquet(fname, output)

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year: {"sumgenweight": sumgenweight, "cutflows": self.cutflows},
            }
        }

    def postprocess(self, accumulator):
        return accumulator
