import numpy as np
from coffea import processor, hist
from functools import partial
import awkward as ak
from boostedhiggs.corrections import (
    add_pileup_weight
)
from coffea.analysis_tools import Weights, PackedSelection 
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
from coffea.nanoevents.methods import vector
import warnings

class HHbbWW(processor.ProcessorABC):
    def __init__(self, year, channel):
        self._year = year
        self._channel = channel
        self._trigger = {
	    2016: {
                "e": [
                    "Ele27_WPTight_Gsf",
                    "Ele45_WPLoose_Gsf",
                    "Ele25_eta2p1_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT350",
                    "Ele15_IsoVVVL_PFHT400",
                    "Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50",
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    ],
                "mu": [
                    "IsoMu24",
                    "IsoTkMu24",
                    "Mu50",
                    "TkMu50",
                    "Mu15_IsoVVVL_PFHT400",
                    "Mu15_IsoVVVL_PFHT350",
	        ],
            },
            2017: {
                "e": [
                    "Ele35_WPTight_Gsf",
                    "Ele32_WPTight_Gsf",
                    "Ele32_WPTight_Gsf_L1DoubleEG",
                    "Ele28_eta2p1_WPTight_Gsf_HT150",
                    "Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT450",
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    ],
                "mu": [
                    "IsoMu27",
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT450",
                ],
	    },
            2018: {
                "e": [
                    "Ele32_WPTight_Gsf",
                    "Ele28_eta2p1_WPTight_Gsf_HT150",
                    "Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT450",
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    ],
                "mu": [
                    "IsoMu24",
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT450",
                ],
            }
        }
        self._trigger = self._trigger[int(self._year)]

        # TODO: check if they vary accross years
        self._metfilters = ["goodVertices",
                            "globalSuperTightHalo2016Filter",
                            "HBHENoiseFilter",
                            "HBHENoiseIsoFilter",
                            "EcalDeadCellTriggerPrimitiveFilter",
                            "BadPFMuonFilter",
                            ]
        
        self._accumulator = processor.dict_accumulator({
            'sumw': processor.defaultdict_accumulator(float),
            'cutflow_SR' : processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            })
        
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):

        # meta infos
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        n_events = len(events)
        selection = PackedSelection()
        weights = Weights(n_events)
        output = self.accumulator.identity()
        if(len(events) == 0): return output

        # gen-weights
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)
        
        # trigger
        triggers = {}
        for channel in ["e","mu"]:
            trigger = np.zeros(len(events), dtype='bool')
            if isRealData: # apply triggers only for data?
                for t in self._trigger[channel]:
                    try:
                        trigger = trigger | events.HLT[t]
                    except:
                        warnings.warn("Missing trigger %s" % t, RuntimeWarning)
            else:
                triggers[channel] = np.ones(len(events), dtype='bool')
            
        # met filter
        met_filters = np.ones(len(events), dtype='bool')
        for t in self._metfilters:
            met_filters = met_filters & events.Flag[t]
        
        # load objects
        muons = events.Muon
        electrons = events.Electron
        jets = events.Jet
        fatjets = events.FatJet
        subjets = events.SubJet
        fatjetsLS = events.FatJetLS
        met = events.MET
        
        # muons
        goodmuon = (
            (muons.mediumId)
            & (muons.miniPFRelIso_all <= 0.2)
            & (muons.pt >= 27)
            & (abs(muons.eta) <= 2.4)
            & (abs(muons.dz) < 0.1)
            & (abs(muons.dxy) < 0.05)
            & (muons.sip3d < 4)
        )
        good_muons = muons[goodmuon]
        ngood_muons = ak.sum(goodmuon, axis=1)

        # electrons
        goodelectron = (
            (electrons.mvaFall17V2noIso_WP90)
            & (electrons.pt >= 30)
            & (abs(electrons.eta) <= 1.479)
            & (abs(electrons.dz) < 0.1)
            & (abs(electrons.dxy) < 0.05)
            & (electrons.sip3d < 4)
        )
        good_electrons = electrons[goodelectron]
        ngood_electrons = ak.sum(goodelectron, axis=1)
        
        # good leptons
        good_leptons = ak.concatenate([good_muons, good_electrons], axis=1)
        good_leptons = good_leptons[ak.argsort(good_leptons.pt)]
        
        # lepton candidate
        candidatelep = ak.firsts(good_leptons)
                
        # jets
        ht = ak.sum(jets[jets.pt > 30].pt,axis=1)
        goodjet = (
            (jets.isTight)
            & (jets.pt > 30)
            & (abs(jets.eta) <= 2.5)
            )
        good_jets = jets[goodjet]

        # fat jets
        # TODO: add soft-drop mass correction
        # TODO: require 2 subjets w. pT>=20 and eta<=2.4? this can probably be done w FatJet_subJetIdx1 or FatJet_subJetIdx2 
        good_fatjet = (
            (fatjets.isTight)
            & (abs(fatjets.eta) <= 2.4)
            & (fatjets.pt > 50)
            & (fatjets.msoftdrop > 30)
            & (fatjets.msoftdrop < 210)
        )
        good_fatjets = fatjets[good_fatjet]

        # hbb candidate
        mask_hbb = (
            (good_fatjets.pt > 200)
            & (good_fatjets.delta_r(candidatelep) > 2.0)
            )
        candidateHbb = ak.firsts(good_fatjets[mask_hbb])
        
        # number of AK4 away from bb jet
        jets_awayHbb = jets[good_jets.delta_r(candidateHbb) >= 1.2]

        # fat jets Lepton Subtracted
        # TODO: add ID
        # TODO: add 2 subjets w pt > 20 & eta<2.4 
        good_fatjetLS = (
            (fatjetsLS.pt > 50)
            & (fatjetsLS.delta_r(candidatelep) > 1.2)
            )
        good_fatjetLSs = fatjetsLS[good_fatjetLS]
        
        # wqq candidate
        mask_hww = (
            (good_fatjetLSs.mass > 10)
            )
        candidateWjj = ak.firsts(good_fatjetLSs[mask_hww][ak.argmin(good_fatjetLSs[mask_hww].delta_r(candidatelep),axis=1,keepdims=True)])

        # TODO: add lvqq likelihood and HWW mass reconstruction
        # For now, reconstruct the mass by taking qq jet, lepton and MET and solving for the z component of the neutrino momentum
        # by requiring that the invariant mass of the group of objects is the Higgs mass = 125
        def getNeutrinoZ(vis,inv,h_mass=125):
            a = h_mass*h_mass - vis.mass*vis.mass + 2*vis.x*inv.x + 2*vis.y*inv.y
            A = 4*(vis.t*vis.t - vis.z*vis.z)
            B = -4*a*vis.z
            C = 4*vis.t*vis.t*(inv.x*inv.x + inv.y*inv.y) - a*a
            delta = B*B - 4*A*C
            invZ = ((delta<0)*( -B/(2*A) )
                   + (delta>0)*( np.maximum( (-B + np.sqrt(delta))/(2*A), (-B - np.sqrt(delta))/(2*A)) ))
            neutrino =  ak.zip({"x": inv.x,
                                "y": inv.y,
                                "z": invZ,
                                "t": np.sqrt(inv.x*inv.x + inv.y*inv.y + invZ*invZ),
                                },
                               with_name="LorentzVector")
            return neutrino
            
        candidateNeutrino = getNeutrinoZ(candidatelep + candidateWjj, met)

        # hh system
        # TODO: verify HWW reconstruction and add neutrino.
        candidateHH = candidateWjj + candidateHbb #+ candidateNeutrino
        
        cuts = {"e":{"trigger": (triggers["e"]),
                     "met_filter": (met_filters),
                     "ht": (ht>=400),
                     "nleptons": ((ngood_electrons==1) & (ngood_muons==0)),
                     "hbb_btag": (candidateHbb.deepTagMD_ZHbbvsQCD >= 0.8), # here you can change to candidateHbb.particleNetMD_Xbb/(1-candidateHbb.particleNetMD_Xcc-candidateHbb.particleNetMD_Xqq)
                     "hbb_vetobtagaway": (ak.max(jets_awayHbb.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium']), # no AK4 b-tagged jets away from bb jet  
                     "hww_tau21_LP": (candidateWjj.tau2/candidateWjj.tau1 <= 0.75),
                     #"hww_tau21_HP": (candidateWjj.tau2/candidateWjj.tau1 <= 0.45), # high purity category
                     "hh_mass": (candidateHH.mass >= 700),
                     "hh_centrality": (candidateHH.pt/candidateHH.mass >= 0.3),
                     },
                "mu":{"trigger": (triggers["mu"]),
                      "met_filter": (met_filters),
                      "ht": (ht>=400),
                      "nleptons": ((ngood_electrons==0) & (ngood_muons==1)), # not sure if need to require 0 muons
                      "hbb_btag": (candidateHbb.deepTagMD_ZHbbvsQCD >= 0.8),
                      "hbb_vetobtagaway": (ak.max(jets_awayHbb.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium']),
                      "hww_tau21_LP": (candidateWjj.tau2/candidateWjj.tau1 <= 0.75),
                      "hh_mass": (candidateHH.mass >= 700),
                      "hh_centrality": (candidateHH.pt/candidateHH.mass >= 0.3),
                      }
                }

        for name, cut in cuts[self._channel].items():
            selection.add(name, cut)
            
        # TODO: add gen info: true neutrino pt, true jet pt, true lep pt
        # TODO: add trigger weights
        if not isRealData:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)

        # fill cutflow
        allcuts_signal = set()
        output['cutflow_SR'][dataset]['none'] += float(weights.weight().sum())
        for cut in cuts[self._channel]:
            allcuts_signal.add(cut)
            output['cutflow_SR'][dataset][cut] += float(weights.weight()[selection.all(*allcuts_signal)].sum())

        return output
        
    def postprocess(self, accumulator):
        return accumulator
