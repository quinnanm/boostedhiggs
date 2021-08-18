import numpy as np
import awkward as ak
import hist as hist2
import json

from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

from boostedhiggs.corrections import (
    corrected_msoftdrop,
    add_pileup_weight,
)
from boostedhiggs.utils import (
    getParticles,
    match_HWWlepqq,
)

import logging
logger = logging.getLogger(__name__)

class HwwProcessor(processor.ProcessorABC):
    def __init__(self, year="2017", jet_arbitration='met', el_wp="wp80"):
        self._year = year
        self._jet_arbitration = jet_arbitration
        self._el_wp = el_wp
        
        self._triggers = {
            2016: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                ],
                'mu': [
                    "Mu50",
                    "Mu55",
                    "Mu15_IsoVVVL_PFHT600",
                ],
            },
            2017: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                ],
                'mu': [
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT600",
                ],
            },
            2018: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                ],
                'mu': [
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT600",
                ],
            }
        }
        self._triggers = self._triggers[int(self._year)]
        
        self._json_paths = {
            '2016': "data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            '2017': "data/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
            '2018': "data/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        }

        self._metfilters = ["goodVertices",
                            "globalSuperTightHalo2016Filter",
                            "HBHENoiseFilter",
                            "HBHENoiseIsoFilter",
                            "EcalDeadCellTriggerPrimitiveFilter",
                            "BadPFMuonFilter",
                        ]

        # WPs for btagDeepFlavB (UL)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self._btagWPs = {
            '2016': {
                'loose': 0.0614,
                'medium': 0.3093,
                'tight': 0.7221,
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
        }

        self.make_output = lambda: {
            'sumw': 0.,
            'signal_kin': hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([0, 2, 4, 6, 8], name='genflavor', label='gen flavor'),
                hist2.axis.IntCategory([0, 1, 4, 6, 9], name='genHflavor', label='higgs matching'),
                hist2.axis.IntCategory([0, 1, 2, 3, 4], name='nprongs', label='Jet nprongs'),
                hist2.storage.Weight(),
            ),
            "jet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.Regular(30, 200, 1000, name='jetpt', label=r'Jet $p_T$ [GeV]'),
                hist2.axis.Regular(30, 15, 200, name="jetmsd", label="Jet $m_{sd}$ [GeV]"),
                hist2.axis.Regular(25, -20, 0, name="jetrho", label=r"Jet $\rho$"),
                hist2.axis.Regular(20, 0, 1, name="btag", label="Jet btag (opphem)"),
                hist2.storage.Weight(),
            ),
            "lep_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(25, 0, 1, name="lepminiIso", label="lep miniIso"),
                hist2.axis.Regular(25, 0, 1, name="leprelIso", label="lep Rel Iso"),
                hist2.axis.Regular(40, 10, 800, name='lep_pt', label=r'lep $p_T$ [GeV]'),
                hist2.axis.Regular(30, 0, 5, name="deltaR_lepjet", label="$\Delta R(l, Jet)$"),
                hist2.storage.Weight(),
            ),
            "met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(30, 0, 500, name="met", label=r"$p_T^{miss}$ [GeV]"),
                hist2.storage.Weight(),
            ),
            "higgs_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(50, 10, 1000, name='matchedHpt', label=r'matched H $p_T$ [GeV]'),
                hist2.axis.Variable(
                    [10,35,60,85,110,135,160,185,210,235,260,285,310,335,360,385,410,450,490,530,570,615,665,715,765,815,865,915,965],
                    name='genHpt', 
                    label=r'genH $p_T$ [GeV]',
                ),
                hist2.storage.Weight(),
            ),
        }
        
    def process(self, events):
        dataset = events.metadata['dataset']
        selection = PackedSelection()
        isRealData = not hasattr(events, "genWeight")
        nevents = len(events)
        weights = Weights(nevents, storeIndividual=True)
        
        output = self.make_output()
        if not isRealData:
            output['sumw'] = ak.sum(events.genWeight)
            weights.add("genweight", events.genWeight)
            
        # trigger
        for channel in ["e","mu"]:
            if isRealData:
                trigger = np.zeros(len(events), dtype='bool')
                for t in self._triggers[channel]:
                    if t in events.HLT.fields:
                        trigger = trigger | events.HLT[t]
                selection.add('trigger'+channel, trigger)
                del trigger
            else:
                selection.add('trigger'+channel, np.ones(nevents, dtype='bool'))

        # TODO: add lumi masks (based on json files) 

        # MET filters
        met_filters = np.ones(nevents, dtype='bool')
        for mf in self._metfilters:
            met_filters = met_filters & events.Flag[mf]
        selection.add('met_filters', met_filters)

        # muons
        goodmuon = (
            (events.Muon.pt > 25)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.mediumId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        lowptmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.looseId
        )
        nlowptmuons = ak.sum(lowptmuon, axis=1)
        
        # electrons
        goodelectron = (
                (events.Electron.pt > 25)
                & (abs(events.Electron.eta) < 2.5)
        )
        if self._el_wp == "wp80":
            goodelectron = ( goodelectron 
                             & (events.Electron.mvaFall17V2noIso_WP80)
                         )
        elif self._el_wp == "wp90":
            goodelectron = ( goodelectron
                             & (events.Electron.mvaFall17V2noIso_WP90)
                         )
        elif self._el_wp == "wpl":
            goodelectron = ( goodelectron
                             & (events.Electron.mvaFall17V2noIso_WPL)
                         )
        else:
            raise RuntimeError("Unknown working point")
        nelectrons = ak.sum(goodelectron, axis=1)
        lowptelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nlowptelectrons = ak.sum(lowptelectron, axis=1)

        # taus
        goodtau = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEle >= 8)
            & (events.Tau.idAntiMu >= 1)
        )
        ntaus = ak.sum(goodtau, axis=1)
            
        selection.add('onemuon', (nmuons == 1) & (nlowptmuons <= 1) & (nelectrons == 0) & (nlowptelectrons == 0) & (ntaus == 0))
        selection.add('oneelectron', (nelectrons == 1) & (nlowptelectrons <= 1) & (nmuons == 0) & (nlowptmuons == 0) & (ntaus == 0))
            
        # concatenate leptons and select leading one
        goodleptons = ak.concatenate([events.Muon[goodmuon], events.Electron[goodelectron]], axis=1)
        candidatelep = ak.firsts(goodleptons[ak.argsort(goodleptons.pt)])
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
            
        # missing transverse energy
        met = events.MET
        selection.add("met_20", met.pt > 20.)

        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjets["qcdrho"] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        
        candidatefj = fatjets[
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight
        ]
        dphi_met_fj = abs(candidatefj.delta_phi(met))
        dr_lep_fj = candidatefj.delta_r(candidatelep_p4)

        if self._jet_arbitration == 'pt':
            candidatefj = ak.firsts(candidatefj)
        elif self._jet_arbitration == 'met':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dphi_met_fj,axis=1,keepdims=True)])
        elif self._jet_arbitration == 'lep':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dr_lep_fj,axis=1,keepdims=True)])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        selection.add("fjmsd", candidatefj.msdcorr > 15.)
            
        # lepton isolation
        # check pfRelIso04 vs pfRelIso03
        selection.add("mu_iso", ( ((candidatelep.pt < 55.) & (candidatelep.pfRelIso03_all < 0.25)) |
                                  ((candidatelep.pt >= 55.) & (candidatelep.miniPFRelIso_all < 0.1)) ) )
        selection.add("el_iso", ( ((candidatelep.pt < 120.) & (candidatelep.pfRelIso03_all < 0.25)) |
                                  ((candidatelep.pt >= 120.) & (candidatelep.miniPFRelIso_all < 0.1)) ) )
        lep_miniIso = candidatelep.miniPFRelIso_all
        lep_relIso = candidatelep.pfRelIso03_all

        # leptons within fatjet
        lep_in_fj = candidatefj.delta_r(candidatelep_p4) < 0.8
        lep_in_fj = ak.fill_none(lep_in_fj, False)
        selection.add("lep_in_fj", lep_in_fj)
        
        # jets
        jets = events.Jet
        jets = jets[
            (jets.pt > 30) 
            & (abs(jets.eta) < 2.5) 
            & jets.isTight
        ]
        dphi_jet_fj = abs(jets.delta_phi(candidatefj))
        dr_jet_fj = abs(jets.delta_r(candidatefj))
        
        # b-jets
        bjets_ophem = ak.max(jets[dphi_jet_fj > np.pi / 2].btagDeepFlavB, axis=1)
        selection.add("btag_ophem_med", bjets_ophem < self._btagWPs[self._year]['medium'])

        # match HWW semi-lep dataset
        if "HWW" in dataset:
            hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs,matchedH,genH,iswlepton,iswstarlepton = match_HWWlepqq(events.GenPart,candidatefj)
            matchedH_pt = ak.firsts(matchedH.pt)
            genH_pt = ak.firsts(genH.pt)
        else:
            hWWlepqq_flavor = ak.zeros_like(candidatefj.pt) 
            hWWlepqq_matched = ak.zeros_like(candidatefj.pt)
            hWWlepqq_nprongs = ak.zeros_like(candidatefj.pt)
            matchedH = ak.zeros_like(candidatefj.pt)
            matchedH_pt = ak.zeros_like(candidatefj.pt)
            genH = ak.zeros_like(candidatefj.pt)
            genH_pt = ak.zeros_like(candidatefj.pt)
            iswlepton = ak.ones_like(candidatefj.pt, dtype=bool)
            iswstarlepton = ak.ones_like(candidatefj.pt, dtype=bool)
            
        selection.add("iswlepton", iswlepton)
        selection.add("iswstarlepton", iswstarlepton)

        regions = {
            "hadel": ["triggere", "met_filters", "lep_in_fj", "fjmsd", "oneelectron", "el_iso", "btag_ophem_med", "met_20"],
            "hadmu": ["triggermu", "met_filters", "lep_in_fj", "fjmsd", "onemuon", "mu_iso", "btag_ophem_med", "met_20"],
            "noselection": []
        }

        if "HWW" in dataset:
            regions["hadel_iswlepton"] = regions["hadel"] + ["iswlepton"]
            regions["hadel_iswstarlepton"] = regions["hadel"] + ["iswstarlepton"]
            regions["hadmu_iswlepton"] = regions["hadmu"] + ["iswlepton"]
            regions["hadmu_iswstarlepton"] = regions["hadmu"] + ["iswstarlepton"]

        # function to normalize arrays after a cut or selection
        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
        
        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)
            
            output['signal_kin'].fill(
                region=region,
                genflavor=normalize(hWWlepqq_flavor, cut),
                genHflavor=normalize(hWWlepqq_matched, cut),
                nprongs=normalize(hWWlepqq_nprongs, cut),
                weight = weights.weight()[cut],
            )
            output["jet_kin"].fill(
                region=region,
                jetpt=normalize(candidatefj.pt, cut),
                jetmsd=normalize(candidatefj.msdcorr, cut),
                jetrho=normalize(candidatefj.qcdrho, cut),
                btag=normalize(bjets_ophem, cut),
                weight=weights.weight()[cut],
            )
            output['lep_kin'].fill(
                region=region,
                lepminiIso=normalize(lep_miniIso, cut),
                leprelIso=normalize(lep_relIso, cut),
                lep_pt=normalize(candidatelep.pt, cut),
                deltaR_lepjet=normalize(candidatefj.delta_r(candidatelep_p4), cut),
                weight=weights.weight()[cut],
            )
            output["met_kin"].fill(
                region=region,
                met=normalize(met.pt, cut),
                weight=weights.weight()[cut],
            )
            if "HWW" in dataset:
                output['higgs_kin'].fill(
                    region=region,
                    matchedHpt=normalize(matchedH_pt, cut),
                    genHpt=normalize(genH_pt, cut),
                    weight=weights.weight()[cut],
                )
            
        for region in regions:
                fill(region)

        return {dataset: output}
            
    def postprocess(self, accumulator):
        return accumulator
