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

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


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


class HwwProcessor(processor.ProcessorABC):
    def __init__(self, year="2017", yearmod="", channels=["ele", "mu", "had"], output_location="./", folder_name=''):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._output_location = output_location
        self.folder_name = folder_name

        # define variables to save for each channel
        self._skimvars = {
            'ele': [
                "lepton_pt",
                "lep_isolation",
                "met",
                "ht",
                "mt_lep_met",
                "dr_jet_candlep",
            ],
            'mu': [
                "lepton_pt",
                "lep_isolation",
                "met",
                "ht",
                "mt_lep_met",
                "dr_jet_candlep",
                "mu_mvaId"
            ],
            'had': [
                "leadingfj_pt",
                "leadingfj_msoftdrop",
                "secondfj_pt",
                "secondfj_msoftdrop",
                "met",
                "ht",
                "bjets_ophem_leadingfj"
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

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self._btagWPs = {
            '2016preVFP': {
                'loose': 0.0508,
                'medium': 0.2598,
                'tight': 0.6502,
            },
            '2016postVFP': {
                'loose': 0.0480,
                'medium': 0.2489,
                'tight': 0.6377,
            },
            '2017': {
                'loose': 0.0532,
                'medium': 0.3040,
                'tight': 0.7476,
            },
            '2018': {
                'loose': 0.0490,
                'medium': 0.2783,
                'tight': 0.7100,
            },
        }[year + yearmod]

        self.selections = {}
        self.cutflows = {}

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            pq.write_table(table, './outfiles/' + ch + folder_name + '/parquet/' + fname + '.parquet')

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
                trigger = np.zeros(len(events), dtype='bool')
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

        # leading lepton
        goodleptons = ak.concatenate([events.Muon[good_muons], events.Electron[good_electrons]], axis=1)
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]
        candidatelep = ak.firsts(goodleptons)

        # candidate leptons
        candidatelep_p4 = ak.zip(
            {
                "pt": candidatelep.pt,
                "eta": candidatelep.eta,
                "phi": candidatelep.phi,
                "mass": candidatelep.mass,
                "charge": candidatelep.charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # relative isolation
        lep_reliso = candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        # mini isolation
        mu_miso = candidatelep.miniPFRelIso_all
        # MVA-ID
        mu_mvaId = candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)

        # JETS
        goodjets = events.Jet[
            (events.Jet.pt > 30)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
        ]
        ht = ak.sum(goodjets.pt, axis=1)

        # FATJETS
        fatjets = events.FatJet
        fatjets["qcdrho"] = 2 * np.log(fatjets.msoftdrop / fatjets.pt)

        good_fatjets = (
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight
            # & fatjets.puId==7   #### TODO field not found
        )
        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]
        leadingfj = ak.firsts(good_fatjets)
        secondfj = ak.pad_none(good_fatjets, 2, axis=1)[:, 1]

        candidatefj_lep = ak.firsts(good_fatjets[ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)])
        # lepton and fatjet mass
        lep_fj_m = (candidatefj_lep - candidatelep_p4).mass

        dphi_jet_lepfj = abs(goodjets.delta_phi(candidatefj_lep))  # ele and mu
        dphi_jet_leadingfj = abs(goodjets.delta_phi(leadingfj))  # had

        bjets_ophem_lepfj = ak.max(goodjets[dphi_jet_lepfj > np.pi / 2].btagDeepFlavB, axis=1)  # in event, pick highest b score in opposite direction from signal
        bjets_ophem_leadingfj = ak.max(goodjets[dphi_jet_leadingfj > np.pi / 2].btagDeepFlavB, axis=1)

        # deltaR
        dr_jet_candlep = candidatefj_lep.delta_r(candidatelep_p4)

        # MET
        met = events.MET
        mt_lep_met = np.sqrt(
            2. * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # event selections
        self.add_selection(
            name='leptonKin',
            sel=(candidatelep.pt > 30),
            channel=['mu']
        )
        self.add_selection(
            name='oneLepton',
            sel=(n_good_muons == 1) & (n_good_electrons == 0) & (n_loose_electrons == 0),
            channel=['mu']
        )
        self.add_selection('leptonIsolation', sel=(
            ((candidatelep.pt > 30)
             & (candidatelep.pt < 55)
             & (lep_reliso < 0.25)
             )
            | ((candidatelep.pt >= 55)
               & (candidatelep.miniPFRelIso_all < 0.2))
        ), channel=['mu'])
        self.add_selection('leptonInJet', sel=(dr_jet_candlep < 0.8), channel=['mu', 'ele'])
        self.add_selection('ht', sel=(ht > 200), channel=['mu', 'ele'])
        self.add_selection('mt', sel=(mt_lep_met < 100), channel=['mu', 'ele'])
        # self.add_selection(
        #     name='bjet_tag',
        #     sel=(bjets_ophem_lepfj > self._btagWPs["medium"]),
        #     channel=['mu', 'ele']
        # )
        # selections for electrons
        self.add_selection(
            name='leptonKin',
            sel=(candidatelep.pt > 40),
            channel=['ele']
        )
        self.add_selection(
            name='oneLepton',
            sel=(n_good_muons == 0) & (n_loose_muons == 0) & (n_good_electrons == 1),
            channel=['ele']
        )
        self.add_selection('leptonIsolation', sel=(
            ((candidatelep.pt > 30)
             & (candidatelep.pt < 120)
             & (lep_reliso < 0.3)
             )
            | ((candidatelep.pt >= 120)
               & (candidatelep.miniPFRelIso_all < 0.2))
        ), channel=['ele'])

        # had selection
        self.add_selection(
            name='oneFatjet',
            sel=(n_fatjets >= 1) & (n_good_muons == 0) & (n_loose_muons == 0) & (n_good_electrons == 0) & (n_loose_electrons == 0),
            channel=['had']
        )
        self.add_selection(
            name='leadingJet',
            sel=leadingfj.pt > 450,
            channel=['had']
        )
        self.add_selection(
            name='softdrop',
            sel=leadingfj.msoftdrop > 30,
            channel=['had']
        )
        self.add_selection(
            name='qcdrho',
            sel=(leadingfj.qcdrho > -7) & (leadingfj.qcdrho < -2.0),
            channel=['had']
        )
#         self.add_selection(
#             name='bjet_tag',
#             sel=(bjets_ophem_leadingfj > self._btagWPs["medium"]),
#             channel=['had']
#         )

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            out = {}
            for var in self._skimvars[ch]:
                if var == "lepton_pt":
                    value = pad_val(candidatelep.pt, -1)
                    out[var] = value
                if var == "dr_jet_candlep":
                    value = pad_val(dr_jet_candlep, -1)
                    out[var] = value
                if var == "mt_lep_met":
                    value = pad_val(mt_lep_met, -1)
                    out[var] = value
                if var == "ht":
                    value = pad_val(ht, -1)
                    out[var] = value
                if var == "met":
                    value = pad_val(met.pt, -1)
                    out[var] = value
                if var == "lep_isolation":
                    value = pad_val(lep_reliso, -1)
                    out[var] = value
                if var == "lepfj_m":
                    value = pad_val(lep_fj_m, -1)
                    out[var] = value
                if var == "candidatefj_lep_pt":
                    value = pad_val(candidatefj_lep.pt, -1)
                    out[var] = value
                if var == "leadingfj_pt":
                    value = pad_val(leadingfj.pt, -1)
                    out[var] = value
                if var == "leadingfj_msoftdrop":
                    value = pad_val(leadingfj.msoftdrop, -1)
                    out[var] = value
                if var == "secondfj_pt":
                    value = pad_val(secondfj.pt, -1)
                    out[var] = value
                if var == "secondfj_msoftdrop":
                    value = pad_val(secondfj.msoftdrop, -1)
                    out[var] = value
                if var == "bjets_ophem_lepfj":
                    value = pad_val(bjets_ophem_lepfj, -1)
                    out[var] = value
                if var == "bjets_ophem_leadingfj":
                    value = pad_val(bjets_ophem_leadingfj, -1)
                    out[var] = value
                else:
                    continue

            # print arrays and selections to debug
            # print(out)
            # print(selections[ch].all(*selections[ch].names))

            # apply selections
            if np.sum(self.selections[ch].all(*self.selections[ch].names)) > 0:
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
        for ch in self._channels:
            if not os.path.exists('./outfiles/' + ch):  # creating a directory for each channel
                os.makedirs('./outfiles/' + ch)
            if not os.path.exists('./outfiles/' + ch + self.folder_name + '/parquet'):  # creating a directory for each channel
                os.makedirs('./outfiles/' + ch + self.folder_name + '/parquet')

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
