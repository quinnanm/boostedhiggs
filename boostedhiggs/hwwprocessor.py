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

from boostedhiggs.utils import match_HWW, getParticles, match_V, match_Top
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
np.seterr(invalid='ignore')


def dsum(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


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
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
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


class HwwProcessor(processor.ProcessorABC):
    def __init__(self,
                 year="2017",
                 yearmod="",
                 channels=["ele", "mu"],
                 output_location="./outfiles/",
                 inference=False,
                 apply_trigger=True):

        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, 'r') as f:
                self._HLTs = json.load(f)[self._year]

        # apply trigger in selection?
        self.apply_trigger = apply_trigger

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, 'r') as f:
                self._metfilters = json.load(f)[self._year]

        # b-tagging corrector
        self._btagWPs = btagWPs["deepJet"][year + yearmod]
        # self._btagSF = BTagCorrector("M", "deepJet", year, yearmod)

        if year == '2018':
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
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:     # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + '/parquet/' + fname + '.parquet')

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: list = None):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        # channels = channel if (channel and channel in self._channels) else self._channels
        channels = channel if channel else self._channels
        for ch in channels:
            self.selections[ch].add(name, sel)
            self.cutflows[ch][name] = np.sum(self.selections[ch].all(*self.selections[ch].names))

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata['dataset']
        isMC = hasattr(events, "genWeight")
        sumgenweight = ak.sum(events.genWeight) if isMC else 0
        nevents = len(events)

        # empty selections and cutflows
        self.selections = {}
        self.cutflows = {}
        for ch in self._channels:
            self.selections[ch] = PackedSelection()
            self.cutflows[ch] = {}
            self.cutflows[ch]["all"] = nevents

        # trigger
        trigger_noiso = {}
        trigger_iso = {}
        for ch in self._channels:
            # apply trigger to both data and MC
            trigger = np.zeros(nevents, dtype='bool')
            trigger_noiso[ch] = np.zeros(nevents, dtype='bool')
            trigger_iso[ch] = np.zeros(nevents, dtype='bool')
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    if "Iso" in t or "WPTight_Gsf" in t:
                        trigger_iso[ch] = trigger_iso[ch] | events.HLT[t]
                    else:
                        trigger_noiso[ch] = trigger_noiso[ch] | events.HLT[t]
                    trigger = trigger | events.HLT[t]
            if self.apply_trigger:
                # apply trigger selection
                self.add_selection("trigger", trigger, [ch])
            del trigger

        # metfilters
        metfilters = np.ones(nevents, dtype='bool')
        metfilterkey = "mc" if isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        self.add_selection("metfilters", metfilters)

        # taus (will need to refine to avoid overlap with htt)
        loose_taus_mu = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        )
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        # muons
        loose_muons = (
            (((events.Muon.pt > 30) & (events.Muon.pfRelIso04_all < 0.25)) |
             (events.Muon.pt > 55))
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (events.Muon.pt > 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.1)
            & (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.sip3d <= 4.0)
            & events.Muon.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # electrons
        loose_electrons = (
            (((events.Electron.pt > 38) & (events.Electron.pfRelIso03_all < 0.25)) |
             (events.Electron.pt > 120))
            & (np.abs(events.Electron.eta) < 2.4)
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            (events.Electron.pt > 38)
            & (np.abs(events.Electron.eta) < 2.4)
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (np.abs(events.Electron.dz) < 0.1)
            & (np.abs(events.Electron.dxy) < 0.05)
            & (events.Electron.sip3d <= 4.0)
            & (events.Electron.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # get candidate lepton
        goodleptons = ak.concatenate([events.Muon[good_muons], events.Electron[good_electrons]], axis=1)    # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]      # sort by pt
        candidatelep = ak.firsts(goodleptons)   # pick highest pt

        candidatelep_p4 = build_p4(candidatelep)    # build p4 for candidate lepton
        lep_reliso = candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all    # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all    # miniso for candidate lepton
        mu_mvaId = candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)      # MVA-ID for candidate lepton
        mu_highPtId = ak.firsts(events.Muon[good_muons]).highPtId
        ele_highPtId = ak.firsts(events.Electron[good_electrons]).cutBased_HEEP

        # jets
        goodjets = events.Jet[
            (events.Jet.pt > 30)
            & (abs(events.Jet.eta) < 5.0)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        ]
        # reject EE noisy jets for 2017
        if self._year == '2017':
            goodjets = goodjets[
                (goodjets.pt > 50)
                | (abs(goodjets.eta) < 2.65)
                | (abs(goodjets.eta) > 3.139)
            ]
        ht = ak.sum(goodjets.pt, axis=1)

        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjets["qcdrho"] = 2 * np.log(fatjets.msdcorr / fatjets.pt)

        good_fatjets = (
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight
        )
        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]        # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]       # sort them by pt

        # for leptonic channel: first clean jets and leptons by removing overlap, then pick candidate_fj closest to the lepton
        lep_in_fj_overlap_bool = good_fatjets.delta_r(candidatelep_p4) > 0.1
        good_fatjets = good_fatjets[lep_in_fj_overlap_bool]
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # MET
        met = events.MET
        mt_lep_met = np.sqrt(
            2. * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )
        # delta phi MET and higgs candidate
        met_fjlep_dphi = candidatefj.delta_phi(met)

        # for leptonic channel: pick candidate_fj closest to the MET
        # candidatefj = ak.firsts(good_fatjets[ak.argmin(good_fatjets.delta_phi(met), axis=1, keepdims=True)])      # get candidatefj for leptonic channel

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
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass
        # to optimize
        # isvbf = ((deta > 3.5) & (mjj > 1000))
        # isvbf = ak.fill_none(isvbf,False)

        """
        HEM issue: Hadronic calorimeter Endcaps Minus (HEM) issue.
        The endcaps of the hadron calorimeter failed to cover the phase space at -3 < eta < -1.3 and -1.57 < phi < -0.87 during the 2018 data C and D.
        The transverse momentum of the jets in this region is typically under-measured, this resuls in over-measured MET. It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = (
                events.run >= 319077 &
                ak.any((
                    (events.Jet.pt > 30.)
                    & (events.Jet.eta > -3.2)
                    & (events.Jet.eta < -1.3)
                    & (events.Jet.phi > -1.57)
                    & (events.Jet.phi < -0.87)
                ), -1)
                | (
                    (met.phi > -1.62)
                    & (met.pt < 470.)
                    & (met.phi < -0.62)
                )
            )

        # event selections for semi-leptonic channels
        self.add_selection(
            name='fatjetKin',
            sel=candidatefj.pt > 200,
            channel=['mu', 'ele']
        )
        self.add_selection(
            name='ht',
            sel=(ht > 200),
            channel=['mu', 'ele']
        )
        # self.add_selection(
        #     name='mt',
        #     sel=(mt_lep_met < 100),
        #     channel=['mu', 'ele']
        # )
        self.add_selection(
            name='antibjettag',
            sel=(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1) < self._btagWPs["M"]),
            channel=['mu', 'ele']
        )
        self.add_selection(
            name='leptonInJet',
            sel=(lep_fj_dr < 0.8),
            channel=['mu', 'ele']
        )

        # event selection for muon channel
        self.add_selection(
            name='leptonKin',
            sel=(candidatelep.pt > 30),
            channel=['mu']
        )
        self.add_selection(
            name='oneLepton',
            sel=(n_good_muons == 1) & (n_good_electrons == 0) & (n_loose_electrons == 0) & ~ak.any(loose_muons & ~good_muons, 1),
            channel=['mu']
        )
        self.add_selection(
            name='notaus',
            sel=(n_loose_taus_mu == 0),
            channel=['mu']
        )
        # self.add_selection(
        #     name='leptonIsolation',
        #     sel=( (candidatelep.pt > 30) & (candidatelep.pt < 55) & (lep_reliso < 0.15) ) | (candidatelep.pt >= 55),
        #     channel=['mu']
        # )
        # self.add_selection(
        #     name='leptonMiniIsolation',
        #     sel=( (candidatelep.pt >= 55) & (candidatelep.miniPFRelIso_all < 0.1) ) | (candidatelep.pt < 55),
        #     # sel= (candidatelep.miniPFRelIso_all < 0.1),
        #     channel=['mu']
        # )

        # event selections for electron channel
        self.add_selection(
            name='leptonKin',
            sel=(candidatelep.pt > 40),
            channel=['ele']
        )
        self.add_selection(
            name='oneLepton',
            sel=(n_good_muons == 0) & (n_loose_muons == 0) & (n_good_electrons == 1) & ~ak.any(loose_electrons & ~good_electrons, 1),
            channel=['ele']
        )
        self.add_selection(
            name='notaus',
            sel=(n_loose_taus_ele == 0),
            channel=['ele']
        )
        # self.add_selection(
        #     name='leptonIsolation',
        #     sel=( (candidatelep.pt > 30) & (candidatelep.pt < 120) & (lep_reliso < 0.15) ) | (candidatelep.pt >= 120),
        #     channel=['ele']
        # )

        # fill tuple variables
        variables = {
            "lep": {
                "fj_pt": candidatefj.pt,
                "fj_msoftdrop": candidatefj.msdcorr,
                "fj_bjets_ophem": ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1),
                "lep_pt": candidatelep.pt,
                "lep_isolation": lep_reliso,
                "lep_misolation": lep_miso,
                "lep_fj_m": lep_fj_m,
                "lep_fj_dr": lep_fj_dr,
                "lep_met_mt": mt_lep_met,
                "met_fj_dphi": met_fjlep_dphi,
            },
            "ele": {
                "ele_highPtId": ele_highPtId,
            },
            "mu": {
                "mu_mvaId": mu_mvaId,
                "mu_highPtId": mu_highPtId,
            },
            "common": {
                "met": met.pt,
                "ht": ht,
                "nfj": n_fatjets,
                "deta": deta,
                "mjj": mjj,
            },
        }

        # gen matching for signal
        if (('HToWW' or 'HWW') in dataset) and isMC:
            matchHWW = match_HWW(events.GenPart, candidatefj)
            variables["lep"]["gen_Hpt"] = ak.firsts(matchHWW["matchedH"].pt)
            variables["lep"]["gen_Hnprongs"] = matchHWW["hWW_nprongs"]
            variables["lep"]["gen_iswlepton"] = matchHWW["iswlepton"]
            variables["lep"]["gen_iswstarlepton"] = matchHWW["iswstarlepton"]

        # gen matching for background
        if ('WJets' in dataset) or ('ZJets' in dataset) and isMC:
            matchV = match_V(events.GenPart, candidatefj)
            if ('WJetsToLNu' in dataset):
                variables["lep"]["gen_isVlep"] = matchV["gen_isVlep"]
            if ('WJetsToQQ' in dataset) or ('ZJetsToQQ' in dataset):
                variables["lep"]["gen_isVqq"] = matchV["gen_isVqq"]
        if ('TT' in dataset) and isMC:
            matchT = match_Top(events.GenPart, candidatefj)
            variables["lep"]["gen_isTop"] = matchT["gen_isTopbmerged"]
            variables["lep"]["gen_isToplep"] = matchT["gen_isToplep"]
            variables["lep"]["gen_isTopqq"] = matchT["gen_isTopqq"]

        # if trigger is not applied then save the trigger variables
        if not self.apply_trigger:
            variables["lep"]["cut_trigger_iso"] = trigger_iso[ch]
            variables["lep"]["cut_trigger_noniso"] = trigger_noiso[ch]

        # let's save the hem veto as a cut for now
        if self._year == "2018":
            variables["common"]["hem_cleaning"] = hem_cleaning

        """
        Weights
        ------
        - Gen weight (DONE)
        - Pileup weight (DONE)
        - L1 prefiring weight for 2016/2017 (DONE)
        - B-tagging efficiency weights (ToDo)
        - Electron trigger scale factors (DONE)
        - Muon trigger scale factors (DONE)
        - Electron ID scale factors and Reco scale factors (DONE)
        - Muon ID scale factors (DONE)
        - Muon Isolation scale factors (DONE)
        - Electron Isolation scale factors (ToDo)
        - Jet Mass Scale (JMS) scale factor (ToDo)
        - Jet Mass Resolution (JMR) scale factor (ToDo)
        - NLO EWK scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq) (DONE)
        - ~NNLO QCD scale factors for DY(ll)/W(lnu)/W(qq)/Z(qq) (DONE)
        - LHE scale weights for signal
        - LHE pdf weights for signal
        - PSweights for signal
        - ParticleNet tagger efficiency
        Up and Down Variations (systematics included as a new variable)
        ----
        - Pileup weight Up/Down (DONE)
        - L1 prefiring weight Up/Down (DONE)
        - B-tagging efficiency Up/Down (ToDo)
        - Electron Trigger Up/Down (ToDo)
        - Muon Trigger Up/Down (DONE)
        - Electron ID Up/Down (DONE)
        - Electron Isolation Up/Down
        - Muon ID Up/Down (DONE)
        - Muon Isolation Up/Down (DONE)
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
        if isMC:
            weights = Weights(nevents, storeIndividual=True)
            weights.add('genweight', events.genWeight)
            if self._year in ("2016", "2017"):
                weights.add("L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
            add_pileup_weight(weights, self._year, self._yearmod, nPU=ak.to_numpy(events.Pileup.nPU))

            add_lepton_weight(weights, candidatelep, self._year + self._yearmod, "muon")
            add_lepton_weight(weights, candidatelep, self._year + self._yearmod, "electron")

            # self._btagSF.addBtagWeight(bjets_away_lepfj, weights, "lep")

            add_VJets_kFactors(weights, events.GenPart, dataset)

            # store the final common weight
            variables["common"]["weight"] = weights.partial_weight(["genweight", "L1Prefiring", "pileup"])

            weights_per_ch = {"ele": [], "mu": []}
            for key in weights._weights.keys():
                # ignore btagSFlight/bc for now
                if "btagSFlight" in key or "btagSFbc" in key:
                    continue

                if "muon" in key:
                    varkey = "mu"
                elif "electron" in key:
                    varkey = "ele"
                elif "lep" in key:
                    varkey = "lep"
                else:
                    varkey = "common"
                # store the individual weights (ONLY for now until we debug)
                variables[varkey][f"weight_{key}"] = weights.partial_weight([key])
                if varkey in weights_per_ch.keys():
                    weights_per_ch[varkey].append(key)

            # store the per channel weight
            for ch in weights_per_ch.keys():
                if len(weights_per_ch[ch]) > 0:
                    variables[ch][f"weight_{ch}"] = weights.partial_weight(weights_per_ch[ch])

            # NOTE: to add variations:
            # for var in weights.variations:
            #     variables["common"][f"weight_{key}"] = weights.weight(key)

        # initialize pandas dataframe
        output = {}

        for ch in self._channels:
            fill_output = True
            # for data, only fill output for the dataset needed
            if not isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                keys = ["common", ch]
                if ch == "ele" or ch == "mu":
                    keys += ["lep"]

                out = {}
                for key in keys:
                    for var, item in variables[key].items():
                        # pad all the variables that are not a cut with -1
                        pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                        # fill out dictionary
                        out[var] = item

                # fill the output dictionary after selections
                output[ch] = {
                    key: value[selection_ch] for (key, value) in out.items()
                }

                # fill inference
                if self.inference:
                    print("pre-inference")
                    pnet_vars = runInferenceTriton(
                        self.tagger_resources_path,
                        events[selection_ch],
                        fj_idx_lep[selection_ch]
                    )
                    print("post-inference")
                    print(pnet_vars)
                    output[ch] = {
                        **output[ch],
                        **{key: value for (key, value) in pnet_vars.items()}
                    }
            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = 'condor_' + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + '/parquet'):
                os.makedirs(self._output_location + ch + '/parquet')

            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {
                'mc': isMC,
                self._year: {
                    'sumgenweight': sumgenweight,
                    'cutflows': self.cutflows
                }
            }
        }

    def postprocess(self, accumulator):
        return accumulator
