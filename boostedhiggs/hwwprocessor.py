import logging
import numpy as np
import awkward as ak
import json
import copy
from coffea import processor, hist
import hist as hist2
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.nanoevents.methods import vector

from boostedhiggs.btag import BTagEfficiency
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    lumiMasks,
)

logger = logging.getLogger(__name__)

class HwwProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt'
             ):
        self._year = year
        self._jet_arbitration = jet_arbitration
        
        self._triggers = {
            2016: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                    "PFHT800",
                    "PFHT900",
                    "AK8PFJet360_TrimMass30",
                    "AK8PFHT700_TrimR0p1PT0p03Mass50",
                    "PFHT650_WideJetMJJ950DEtaJJ1p5",
                    "PFHT650_WideJetMJJ900DEtaJJ1p5",
                    "PFJet450",
                ],
                'mu': [
                    "Mu50",
                    "Mu55",
                    "Mu15_IsoVVVL_PFHT600",
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
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                    ],
                'mu': [
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT600",
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                ],
                    },
            2018: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFHT1050",
                    "PFJet500",
                    "AK8PFJet500",
                    ],
                'mu': [
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT600",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFHT1050",
                    "PFJet500",
                    "AK8PFJet500",
                ],
            }
        }
        self._triggers = self._triggers[int(self._year)]
        
        self._json_paths = {
            '2016': "jsons/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt", 
            '2017': "jsons/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt",
            '2018': "jsons/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt",
        }

        self._metfilters = ["goodVertices",
                            "globalSuperTightHalo2016Filter",
                            "HBHENoiseFilter",
                            "HBHENoiseIsoFilter",
                            "EcalDeadCellTriggerPrimitiveFilter",
                            "BadPFMuonFilter",
                        ]

        def process(self, events):
            isRealData = not hasattr(events, "genWeight")
            
            # for now, always run w no shift (a.k.a. jet systematic)
            return self.process_shift(events, None)

        def process_shift(self, events, shift_name):
            dataset = events.metadata['dataset']
            isRealData = not hasattr(events, "genWeight")
            selection = PackedSelection()
            weights = Weights(len(events), storeIndividual=True)
            output = self.make_output()
            if shift_name is None and not isRealData:
                output['sumw'] = ak.sum(events.genWeight)

            # trigger
            triggers = {}
            for channel in ["e","mu"]:
                if isRealData:
                    trigger = np.zeros(len(events), dtype='bool')
                    for t in self._triggers[channel]:
                        if t in events.HLT.fields:
                            trigger = trigger | events.HLT[t]
                    selection.add('trigger'+channel, trigger)
                    del trigger
                else:
                    selection.add('trigger'+channel, np.ones(len(events), dtype='bool'))

            # lumi mask
            if isRealData:
                selection.add('lumimask', lumiMasks[self._year](events.run, events.luminosityBlock))
            else:
                selection.add('lumimask', np.ones(len(events), dtype='bool'))
                
            # met filters
            met_filters = np.ones(len(events), dtype='bool')
            for t in self._metfilters:
                met_filters = met_filters & events.Flag[t]
            selection.add('metfilters', met_filters)
        
            # met
            met = events.MET
            selection.add('met50p', met.pt > 50.)
            selection.add('met150p', met.pt > 150.)

            # leptons
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
            
            goodelectron = (
                (events.Electron.pt > 25)
                & (abs(events.Electron.eta) < 2.5)
                & (events.Electron.mvaFall17V2noIso_WP80)
            )
            nelectrons = ak.sum(goodelectron, axis=1)
            lowptelectron = (
                (events.Electron.pt > 10)
                & (abs(events.Electron.eta) < 2.5)
                & (events.Electron.cutBased >= events.Electron.LOOSE)
            )
            nlowptelectrons = ak.sum(lowptelectron, axis=1)
            
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

            selection.add('lep_miniIso', candidatelep.miniPFRelIso_all < 0.1)
            selection.add('muon_kin', (candidatelep.pt > 25.) & (abs(candidatelep.eta) < 2.1))
            selection.add('electron_kin', (candidatelep.pt > 25.) & (abs(candidatelep.eta) < 2.1))

            # taus
            if self._year=='2018':
                tauAntiEleId = events.Tau.idAntiEle2018
            else:
                tauAntiEleId = events.Tau.idAntiEle
            goodtau = (
                (events.Tau.pt > 20)
                & (abs(events.Tau.eta) < 2.3)
                & (tauAntiEleId >= 8)
                & (events.Tau.idAntiMu >= 1)
            )
            ntaus = ak.sum(goodtau, axis=1)

            selection.add('onemuon', (nmuons == 1) & (nlowptmuons <= 1) & (nelectrons == 0) & (nlowptelectrons == 0) & (ntaus == 0))
            selection.add('oneelectron', (nelectrons == 1) & (nlowptelectrons <= 1) & (nmuons == 0) & (nlowptmuons == 0) & (ntaus == 0))

            mt_lepmet = np.sqrt(2.*candidatelep.pt*met.pt*(ak.ones_like(candidatelep.pt) - np.cos(candidatelep_p4.delta_phi(met))))
            selection.add('mt_lepmet_80m', (mt_lepmet < 80.))
            selection.add('mt_lepmet_80p', (mt_lepmet >= 80.))

            # fatjets
            fatjets = events.FatJet
            fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
            fatjets['qcdrho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
            
            candidatefj = fatjets[
                (fatjets.pt > 200)
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

            selection.add('fj_minkin',
                          (candidatefj.pt > 300)
                          & (candidatefj.msdcorr > 40.)
                          & (abs(candidatefj.eta) < 2.4)
                          & (candidatefj.qcdrho > -6.)
                          & (candidatefj.qcdrho < -1.75)
                      )
            selection.add('fj_id', candidatefj.isTight)

            selection.add('taufj_dr',ak.max(events.Tau[goodtau].delta_r(candidatefj))<0.8)
            selection.add('lepfj_dr',ak.max(candidatefj.delta_r(candidatelep_p4))<0.8)
            selection.add('metfj_dphi',ak.max(abs(candidatefj.delta_phi(met)))<2*np.pi/3)
            
            # jets
            jets = events.Jet
            jets = jets[
                (jets.pt > 30.)
                & (abs(jets.eta) < 2.5)
                & jets.isTight
            ][:, :4] # only consider first four jets
            dphi_jet_fj = abs(jets.delta_phi(candidatefj))
            dr_jet_fj = abs(jets.delta_r(candidatefj))

            # b-jets
            # TODO: switch from btagDeepB to btagDeepFlavC
            selection.add('antibtag_opphem', ak.max(jets[dphi_jet_fj > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
            selection.add('btag_opphem', ak.max(jets[dphi_jet_fj > np.pi / 2].btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])
            selection.add('btag_away', ak.max(jets[dr_jet_fj > 0.8].btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

            if isRealData :
                genflavor = ak.zeros_like(candidatejet.pt)
            else:       
                weights.add('genweight', events.genWeight)
                add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
                logger.debug("Weight statistics: %r" % weights.weightStatistics)

            regions = {
                'hadmu_signal': ['fj_minkin', 'triggermu', 'fj_id', 'antibtag_opphem', 'met50p', 'onemuon', 'muon_kin', 'lepfj_dr', 'taufj_dr', 'mt_lepmet_80m', 'lep_miniIso'],
                'hadel_signal': ['fj_minkin', 'triggerel', 'fj_id', 'antibtag_opphem', 'met50p', 'oneelectron', 'electron_kin', 'lepDrAK8', 'taufj_dr', 'mt_lepmet_80m', 'lep_miniIso'],
            }

            def normalize(val, cut):
                if cut is None:
                    ar = ak.to_numpy(ak.fill_none(val, np.nan))
                    return ar
                else:
                    ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                    return ar

        def postprocess(self, accumulator):
            return accumulator
