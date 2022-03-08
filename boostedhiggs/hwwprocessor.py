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

from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection
from boostedhiggs.utils import match_HWW
from boostedhiggs.btag import btagWPs
from boostedhiggs.btag import BTagCorrector

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    def __init__(self, year="2017", yearmod="", channels=["ele", "mu", "had"], output_location="./outfiles/", append="", local=False):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._output_location = output_location
        self.append = append
        self.local = local

        # define variables to save for each channel
        self._skimvars = {
            'ele': [
                "lep_pt",
                "lep_isolation",
                "lep_misolation",
                "lep_fj_m",
                "lep_fj_bjets_ophem",
                "lep_fj_dr",
                "fj_msoftdrop",
                "fj_pt",
                "lep_met_mt",
                "met",
                "ht",
                "lep_matchedH",
                "lep_nprongs",
                "lep_iswlepton",
                "lep_iswstarlepton",
                "weight",
            ],
            'mu': [
                "lep_pt",
                "lep_isolation",
                "lep_misolation",
                "lep_fj_m",
                "lep_fj_bjets_ophem",
                "lep_fj_dr",
                "lep_mvaId",
                "fj_msoftdrop",
                "fj_pt",
                "lep_met_mt",
                "met",
                "ht",
                "lep_matchedH",
                "lep_nprongs",
                "lep_iswlepton",
                "lep_iswstarlepton",
                "weight",
            ],
            'had': [
                "fj0_msoftdrop",
                "fj0_pt",
                "fj0_pnh4q",
                "fj1_msoftdrop",
                "fj1_pt",
                "fj1_pnh4q",
                "fj0_bjets_ophem",
                "met",
                "ht",
                "had_machedH",
                "had_nprongs",
                "weight",
            ],
        }

        # trigger paths
        self._HLTs = {
            2016: {
                'ele': [
                    "Ele27_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon175",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "TkMu50",
                    "IsoMu24",
                    "IsoTkMu24",
                    # "Mu55",
                    # "Mu15_IsoVVVL_PFHT600" # VVL
                ],
                'had': [
                    "PFHT800",
                    "PFHT900",
                    "AK8PFJet360_TrimMass30",
                    "AK8PFHT700_TrimR0p1PT0p03Mass50",
                    "PFHT650_WideJetMJJ950DEtaJJ1p5",
                    "PFHT650_WideJetMJJ900DEtaJJ1p5",
                    "PFJet450",
                ],
            },
            2017: {
                'ele': [
                    "Ele35_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon200",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "IsoMu27",
                    "OldMu100",
                    "TkMu100",
                    # "Mu15_IsoVVVL_PFHT600", # VVL
                ],
                'had': [
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                ],
            },
            2018: {
                'ele': [
                    "Ele32_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon200",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "IsoMu24",
                    "OldMu100",
                    "TkMu100",
                    # "Mu15_IsoVVVL_PFHT600", # VVL
                ],
                'had': [
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                ],
            }
        }[int(self._year)]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        self._metfilters = {
            2016: [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "eeBadScFilter",
            ],
            2017: [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                # "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
            2018:  [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                # "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
        }[int(self._year)]

        self._btagWPs = btagWPs["deepJet"][year + yearmod]
        # self.btagCorr = BTagCorrector("M", "deepJet", year, yearmod)

        self.selections = {}
        self.cutflows = {}

        self.dataset_per_ch = {
            "ele": "SingleElectron",
            "mu": "SingleMuon",
            "had": "JetHT",
        }

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
        triggers = {}
        for ch in self._channels:
            if ch == "had" and isMC:
                trigger = np.ones(nevents, dtype='bool')
            else:
                # apply trigger to both data and MC (except for hadronic channel)
                trigger = np.zeros(nevents, dtype='bool')
                for t in self._HLTs[ch]:
                    if t in events.HLT.fields:
                        trigger = trigger | events.HLT[t]
            self.add_selection("trigger", trigger, [ch])
            del trigger

        # metfilters
        metfilters = np.ones(nevents, dtype='bool')
        for mf in self._metfilters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        self.add_selection("metfilters", metfilters)

        # define muon objects
        loose_muons = (
            (((events.Muon.pt > 30) & (events.Muon.pfRelIso04_all < 0.25)) |
             (events.Muon.pt > 55))
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (events.Muon.pt > 28)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.1)
            & (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.sip3d <= 4.0)
            & events.Muon.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # define electron objects
        loose_electrons = (
            (((events.Electron.pt > 38) & (events.Electron.pfRelIso03_all < 0.25)) |
             (events.Electron.pt > 120))
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            (events.Electron.pt > 38)
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (np.abs(events.Electron.dz) < 0.1)
            & (np.abs(events.Electron.dxy) < 0.05)
            & (events.Electron.sip3d <= 4.0)
            & (events.Electron.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # define tau objects
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

        # leading lepton
        goodleptons = ak.concatenate([events.Muon[good_muons], events.Electron[good_electrons]], axis=1)
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]
        candidatelep = ak.firsts(goodleptons)

        # candidate leptons
        candidatelep_p4 = build_p4(candidatelep)
        # relative isolation
        lep_reliso = candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        # mini isolation
        lep_miso = candidatelep.miniPFRelIso_all
        # MVA-ID
        mu_mvaId = candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)

        # JETS
        goodjets = events.Jet[
            (events.Jet.pt > 30)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        ]
        ht = ak.sum(goodjets.pt, axis=1)

        # FATJETS
        fatjets = events.FatJet
        fatjets["qcdrho"] = 2 * np.log(fatjets.msoftdrop / fatjets.pt)

        good_fatjets = (
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight
        )
        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]
        leadingfj = ak.firsts(good_fatjets)
        secondfj = ak.pad_none(good_fatjets, 2, axis=1)[:, 1]

        # for hadronic channels: leading pt
        candidatefj = leadingfj
        dphi_jet_leadingfj = abs(goodjets.delta_phi(leadingfj))

        # for leptonic channel: leading pt which contains lepton
        candidatefj_lep = ak.firsts(good_fatjets[ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)])
        dphi_jet_lepfj = abs(goodjets.delta_phi(candidatefj_lep))

        # lepton and fatjet mass
        lep_fj_m = (candidatefj_lep - candidatelep_p4).mass  # mass of fatjet without lepton

        # b-jets
        # in event, pick highest b score in opposite direction from signal (we will make cut here to avoid tt background events producing bjets)
        bjets_away_lepfj = goodjets[dphi_jet_lepfj > np.pi / 2]
        bjets_away_leadingfj = goodjets[dphi_jet_leadingfj > np.pi / 2]

        # deltaR
        dr_jet_candlep = candidatefj_lep.delta_r(candidatelep_p4)

        # MET
        met = events.MET
        mt_lep_met = np.sqrt(
            2. * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # event selections for muon channel
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
        #     'leptonIsolation',
        #     sel=(((candidatelep.pt > 30) & (candidatelep.pt < 55) & (lep_reliso < 0.25)) | ((candidatelep.pt >= 55) & (candidatelep.miniPFRelIso_all < 0.2)))
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
        #     'leptonIsolation',
        #     sel=(((candidatelep.pt > 30) & (candidatelep.pt < 120) & (lep_reliso < 0.3)) | ((candidatelep.pt >= 120) & (candidatelep.miniPFRelIso_all < 0.2))),
        #     channel=['ele']
        # )

        # event selections for both leptonic channels
        # self.add_selection(
        #     name='leptonInJet',
        #     sel=(dr_jet_candlep < 0.8),
        #     channel=['mu', 'ele']
        # )
        self.add_selection(
            name='anti_bjettag',
            sel=(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1) < self._btagWPs["M"]),
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

        # event selections for hadronic channel
        self.add_selection(
            name='oneFatjet',
            sel=(n_fatjets >= 1) &
                (n_good_muons == 0) & (n_loose_muons == 0) &
                (n_good_electrons == 0) & (n_loose_electrons == 0),
            channel=['had']
        )
        self.add_selection(
            name='fatjetKin',
            sel=leadingfj.pt > 300,
            channel=['had']
        )
        self.add_selection(
            name='fatjetSoftdrop',
            sel=leadingfj.msoftdrop > 30,
            channel=['had']
        )
        self.add_selection(
            name='qcdrho',
            sel=(leadingfj.qcdrho > -7) & (leadingfj.qcdrho < -2.0),
            channel=['had']
        )
        self.add_selection(
            name='anti_bjettag',
            sel=(ak.max(bjets_away_leadingfj.btagDeepFlavB, axis=1) < self._btagWPs["M"]),
            channel=['had']
        )
        self.add_selection(
            name='met',
            sel=(met.pt < 200),
            channel=['had']
        )

        variables = {}
        variables["lep_pt"] = pad_val(candidatelep.pt, -1)
        variables["lep_isolation"] = pad_val(lep_reliso, -1)
        variables["lep_misolation"] = pad_val(lep_miso, -1)
        variables["lep_fj_m"] = pad_val(lep_fj_m, -1)
        variables["lep_fj_bjets_ophem"] = pad_val(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1), -1)
        variables["lep_fj_dr"] = pad_val(dr_jet_candlep, -1)
        variables["lep_mvaId"] = pad_val(mu_mvaId, -1)
        variables["fj_msoftdrop"] = pad_val(candidatefj_lep.msoftdrop, -1)
        variables["fj_pt"] = pad_val(candidatefj_lep.pt, -1)
        variables["met"] = pad_val(met.pt, -1)
        variables["ht"] = pad_val(ht, -1)
        variables["fj0_msoftdrop"] = pad_val(leadingfj.msoftdrop, -1)
        variables["fj0_pt"] = pad_val(leadingfj.pt, -1)
        variables["fj0_pnh4q"] = pad_val(leadingfj.particleNet_H4qvsQCD, -1)
        variables["fj1_msoftdrop"] = pad_val(secondfj.msoftdrop, -1)
        variables["fj1_pt"] = pad_val(secondfj.pt, -1)
        variables["fj1_pnh4q"] = pad_val(secondfj.particleNet_H4qvsQCD, -1)
        variables["fj0_bjets_ophem"] = pad_val(ak.max(bjets_away_leadingfj.btagDeepFlavB, axis=1), -1)

        # weights
        # TODO:
        # - pileup weight
        # - L1 prefiring weight for 2016/2017
        # - btag weights
        # - electron,muon,ht trigger scale factors
        # - lepton ID scale factors
        # - lepton isolation scale factors
        # - JMS scale factor
        # - JMR scale factor
        # - EWK NLO scale factors for DY(ll)
        # - EWK and QCD scale factors for W/Z(qq)
        # - lhe scale weights for signal
        # - lhe pdf weights for signal
        # - top pt reweighting for top
        # - psweights for signal
        if isMC:
            weights = Weights(nevents, storeIndividual=True)
            weights.add('genweight', events.genWeight)
            # self.btagCorr.addBtagWeight(bjets_away_lepfj, weights)
            # self.btagCorr.addBtagWeight(bjets_away_leadingfj, weights)
            variables["weight"] = pad_val(weights.weight(), -1)

        # systematics
        # - trigger up/down (variable)
        # - btag weights up/down (variable)
        # - l1 prefiring up/down (variable)
        # - pu up/down (variable)
        # - JES up/down systematics (new output files)
        # - JER up/down systematics (new output files)
        # - MET unclustered up/down (new output files)
        # - JMS (variable)
        # - JMR (variable)
        # - tagger SF (variable)

        # higgs matching
        if (('HToWW' or 'HWW') in dataset) and isMC:
            match_HWW_had = match_HWW(events.GenPart, candidatefj)
            match_HWW_lep = match_HWW(events.GenPart, candidatefj_lep)

            variables["had_nprongs"] = pad_val(match_HWW_had["hWW_nprongs"], -1)
            variables["had_matchedH"] = pad_val(ak.firsts(match_HWW_had["matchedH"].pt), -1)

            variables["lep_nprongs"] = pad_val(match_HWW_lep["hWW_nprongs"], -1)
            variables["lep_iswlepton"] = pad_val(match_HWW_lep["iswlepton"], -1)
            variables["lep_iswstarlepton"] = pad_val(match_HWW_lep["iswstarlepton"], -1)
            variables["lep_matchedH"] = pad_val(ak.firsts(match_HWW_lep["matchedH"]).pt, -1)

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            out = {}
            for var in self._skimvars[ch]:
                if var in variables.keys():
                    out[var] = variables[var]

            fill_output = True
            # for data, only fill output for that channel
            if not isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False
            # only fill output for that channel if the selections yield any events
            if np.sum(self.selections[ch].all(*self.selections[ch].names)) <= 0:
                fill_output = False

            if fill_output:
                output[ch] = {
                    key: value[self.selections[ch].all(*self.selections[ch].names)] for (key, value) in out.items()
                }
            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = 'condor_' + fname

        print(self._output_location)
        if self.local:
            if not os.path.exists(self._output_location + dataset):
                os.makedirs(self._output_location + dataset)
                self._output_location = self._output_location + dataset + '/' + append
        else:
            self._output_location = self._output_location + append

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + '/parquet'):
                os.makedirs(self._output_location + ch + '/parquet')

            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {'mc': isMC,
                      self._year: {'sumgenweight': sumgenweight,
                                   'cutflows': self.cutflows}
                      }
        }

    def postprocess(self, accumulator):
        return accumulator
