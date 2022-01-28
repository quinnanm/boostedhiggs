from functools import partial
import numpy as np
import awkward as ak
import hist as hist2
import json
from copy import deepcopy

from coffea import processor, hist
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

from .corrections import (
    corrected_msoftdrop,
#    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_VJets_kFactors,
    add_TopPtReweighting,
#    add_jetTriggerWeight,
#    add_TriggerWeight,
    add_LeptonSFs,
    add_METSFs,
    add_pdf_weight,
    add_scalevar_7pt,
    add_scalevar_3pt,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
)

from .common import (
    getBosons,
    getParticles,
    matchedBosonFlavor,
    matchedBosonFlavorLep,
    getHTauTauDecayInfo,
    isOverlap,
)

import logging
logger = logging.getLogger(__name__)

# function to normalize arrays after a cut or selection
def normalize(val, cut=None):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar

class HttProcessor(processor.ProcessorABC):
    def __init__(self, year="2017", jet_arbitration='met', plotopt=0, yearmod="", skipJER=False):
        self._year = year
        self._yearmod = yearmod
        self._plotopt = plotopt
        self._jet_arbitration = jet_arbitration
        self._skipJER = skipJER
        
        self._triggers = {
            '2016': {
                'e': [
                    "Ele27_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon175",
                ],
                'mu': [
                    "Mu50",
                    "TkMu50",
                    "IsoMu24",
                    "IsoTkMu24",
                ],
                'had': [
                    'PFHT800',
                    'PFHT900',
                    'AK8PFJet360_TrimMass30',
                    'AK8PFHT700_TrimR0p1PT0p03Mass50',
                    'PFHT650_WideJetMJJ950DEtaJJ1p5',
                    'PFHT650_WideJetMJJ900DEtaJJ1p5',
                    'PFJet450',
                ],
                'met': [
                    #"PFMETNoMu120_PFMHTNoMu120_IDTight",
                    "PFMET120_PFMHT120_IDTight",
                ],
            },
            '2017': {
                'e': [
                    'Ele35_WPTight_Gsf',
                    'Ele115_CaloIdVT_GsfTrkIdT',
                    'Photon200',
                ],
                'mu': [
                    "Mu50",
                    "IsoMu27",
                    "OldMu100",
                    "TkMu100",
                ],
                'had': [
                    'PFHT1050',
                    'AK8PFJet400_TrimMass30',
                    'AK8PFJet420_TrimMass30',
                    'AK8PFHT800_TrimMass50',
                    'PFJet500',
                    'AK8PFJet500',
                ],
                'met': [
                    #"PFMETNoMu120_PFMHTNoMu120_IDTight",
                    "PFMET120_PFMHT120_IDTight",
                    "PFMET120_PFMHT120_IDTight_PFHT60",
                ],
            },
            "2018": {
                'e': [
                    'Ele32_WPTight_Gsf',
                    'Ele115_CaloIdVT_GsfTrkIdT',
                    'Photon200',
                ],
                'mu': [
                    "Mu50",
                    "IsoMu24",
                    "OldMu100",
                    "TkMu100",
                ],
                'had': [
                    'PFHT1050',
                    'AK8PFJet400_TrimMass30',
                    'AK8PFJet420_TrimMass30',
                    'AK8PFHT800_TrimMass50',
                    'PFJet500',
                    'AK8PFJet500',
                ],
                'met': [
                    #"PFMETNoMu120_PFMHTNoMu120_IDTight",
                    "PFMET120_PFMHT120_IDTight",
                ],
            }
        }[year]

        self._metFilters = {
            '2016': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
            ],
            '2017': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
            '2018': [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
        }[year]

        # WPs for btagDeepFlavB (UL)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self._btagWPs = {
            '2016preVFP': {
                'medium': 0.6001,
            },
            '2016postVFP': {
                'medium': 0.5847,
            },
            '2017': {
                'medium': 0.4506,
            },
            '2018': {
                'medium': 0.4168,
            },
        }[year+yearmod]

        jet_pt_bin = hist.Bin('jet_pt', r'Jet $p_{T}$ [GeV]', 40, 200., 1200.)
        jet_eta_bin = hist.Bin('jet_eta', r'Jet $\eta$', 20, -3., 3.)
        jet_msd_bin = hist.Bin('jet_msd', r'Jet $m_{sd}$ [GeV]', 50, 0., 500.)
        nn_hadhad_bin = hist.Bin('nn_hadhad',r'$NN_{\tau_{h}\tau_{h}}$', [0.,0.1,0.5,0.8,0.9,0.95,0.99,0.995,0.999,0.9999,0.99995,0.99999,0.999999,1.])
        nn_hadhad_qcd_bin = hist.Bin('nn_hadhad_qcd',r'$NN_{\tau_{h}\tau_{h}}$', [0.,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9,1.])
        nn_hadhad_wjets_bin = hist.Bin('nn_hadhad_wjets',r'$NN_{\tau_{h}\tau_{h}}$', [0.,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9,1.])
        nn_hadel_bin = hist.Bin('nn_hadel',r'$NN_{e\tau_{h}}$', [0.,0.1,0.5,0.8,0.9,0.95,0.99,0.995,0.999,0.9999,0.99995,0.99999,0.999999,1.])
        nn_hadmu_bin = hist.Bin('nn_hadmu',r'$NN_{\mu\tau_{h}}$', [0.,0.1,0.5,0.8,0.9,0.95,0.99,0.995,0.999,0.9999,0.99995,0.99999,0.999999,1.])
        ztagger_mu_qcd_bin = hist.Bin('ztagger_mu_qcd',r'$Z^{\mu\tau}_{NN}[QCD]$', [0.,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9,1.])
        ztagger_mu_mm_bin = hist.Bin('ztagger_mu_mm',r'$Z^{\mu\tau}_{NN}[Z\mu\mu]$', [0.,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9,1.])
        ztagger_mu_hm_bin = hist.Bin('ztagger_mu_hm',r'$Z^{\mu\tau}_{NN}[Z\mu\tau]$', [0.,0.1,0.5,0.8,0.9,0.95,0.99,0.995,0.999,0.9999,0.99995,0.99999,0.999999,1.])
        nn_disc_bin = hist.Bin('nn_disc',r'$NN$', [0.,0.1,0.5,0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.999,0.9995,0.9999,0.99999,0.999999,0.9999999,1.000001])
        massreg_bin = hist.Bin('massreg',r'$m_{NN}$', [0.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,200.,250.,300.,350.,400.,450.,500.])
        massreg_fine_bin = hist.Bin('massreg',r'$m_{NN}$', 100, 0., 500.)
        ztagger_bin = hist.Bin('ztagger', r'$Z^{\ell\tau}_{NN}$', 100, 0., 1.)
        mt_lepmet_bin = hist.Bin('mt_lepmet', r'$m_{T}(\ell, MET)$', 50, 0., 500.)
        mt_jetmet_bin = hist.Bin('mt_jetmet', r'$m_{T}(j, MET)$', 100, 0., 1000.)
        oppbjet_pt_bin = hist.Bin('oppbjet_pt', r'Max opp. deepCSV-bjet $p_{T}$ [GeV]', 20, 0., 500)
        oppbtag_bin = hist.Bin('oppbtag', r'Max opp. deepCSV-b', 10, 0., 1)
        lep_pt_bin = hist.Bin('lep_pt', r'Lepton $p_{T}$ [GeV]', 50, 0., 500.)
        lep_eta_bin = hist.Bin('lep_eta', r'Lepton $\eta$', 20, -3., 3.)
        jet_lsf3_bin = hist.Bin('lsf3', r'Jet LSF$_3$', 20, 0., 1.)
        lep_jet_dr_bin = hist.Bin('lep_jet_dr', r'$\Delta R(jet,lepton)$', 40, 0., 4.)
        lep_miso_bin = hist.Bin('lep_miso', r'Lepton miniIso', 80, 0., 0.4)
        jetlep_m_bin = hist.Bin('jetlep_m', r'Jet-lepton $m$ [GeV]', 20, 0, 200.)
        jetmet_m_bin = hist.Bin('jetmet_m', r'Jet+MET $m$ [GeV]', 20, 0, 600.)
        jetlepmet_m_bin = hist.Bin('jetlepmet_m', r'Jet+lepton+MET $m$ [GeV]', 20, 0, 600.)
        jetmet_dphi_bin = hist.Bin('jetmet_dphi', r'$\Delta\phi(jet,MET)$', 2, 0., 3.14)
        jetmet_dphi_fine_bin = hist.Bin('jetmet_dphi', r'$\Delta\phi(jet,MET)$', 35, 0., 3.5)
        met_pt_bin = hist.Bin('met_pt', r'PuppiMET [GeV]', [20.,50.,75.,100.,150.,1000.])
        met_nopup_pt_bin = hist.Bin('met_nopup_pt', r'MET [GeV]', 100, 0, 1000)
        met_pup_pt_bin = hist.Bin('met_pup_pt', r'PUPPI MET [GeV]', 100, 0, 1000)
        n2b1_bin = hist.Bin('n2b1', r'N_{2}', 2, -1.,1.)
        h_pt_bin = hist.Bin('h_pt', r'h $p_{T}$ [GeV]', [250,280,300,350,400,500,600,1200])
        ntau_bin = hist.Bin('ntau',r'Number of taus',64,-0.5,63.5)
        antilep_bin = hist.Bin('antilep',r'Anti lepton veto',3,-1.5,1.5)
        genhtt_bin = hist.Bin('genhtt',r'hh,eh,mh,em,ee,mm (- for dr > 0.8)',4,-0.5,3.5)
        gentau1had_bin = hist.Bin('gentau1had',r'1pr,1pr+pi0,3pr',4,-0.5,3.5)
        gentau2had_bin = hist.Bin('gentau2had',r'1pr,1pr+pi0,3pr',4,-0.5,3.5)
        met_trigger_bin = hist.Bin('met_trigger',r'Pass MET Trigger', 2,-0.5,1.5)

        self._altplots = (
            { 'jet_pt':jet_pt_bin, 'jet_eta':jet_eta_bin, 'jet_msd':jet_msd_bin, 'mt_lepmet':mt_lepmet_bin, 'mt_jetmet': mt_jetmet_bin, 'lep_pt':lep_pt_bin, 'lep_eta':lep_eta_bin, 'lep_jet_dr':lep_jet_dr_bin, 'n2b1':n2b1_bin, 'jetlep_m':jetlep_m_bin, 'met_nopup_pt':met_nopup_pt_bin, 'met_pup_pt':met_pup_pt_bin, 'jetmet_dphi':jetmet_dphi_fine_bin, 'massreg':massreg_fine_bin, 'ztagger':ztagger_bin},
            { 'lep_miso':lep_miso_bin, 'nn_hadhad':nn_hadhad_bin , 'nn_hadhad_qcd':nn_hadhad_qcd_bin, 'nn_hadhad_wjets':nn_hadhad_wjets_bin, 'ztagger_mu_qcd':ztagger_mu_qcd_bin, 'ztagger_mu_mm':ztagger_mu_mm_bin, 'ztagger_mu_hm':ztagger_mu_hm_bin, 'nn_hadel':nn_hadel_bin, 'nn_hadmu':nn_hadmu_bin, 'antilep':antilep_bin}, 
            {'met_pt':met_pup_pt_bin, 'met_trigger':met_trigger_bin},
        )[self._plotopt-1]

        if self._plotopt==3:
            self._accumulator = processor.dict_accumulator({
                # dataset -> sumw
                'sumw': processor.defaultdict_accumulator(float),
                # dataset -> cut -> count
                'cutflow_hadel_base': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_base': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_lep': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_lep': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_jet': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_jet': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_signal': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_signal': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            })
        else:
            self._accumulator = processor.dict_accumulator({
                # dataset -> sumw
                'sumw': processor.defaultdict_accumulator(float),
                # dataset -> cut -> count
                'cutflow_hadhad_signal': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_signal_met': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_anti_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                #'cutflow_hadhad_cr_b': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_met': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_met_anti_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_met_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                #'cutflow_hadhad_cr_b_mu': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_mu_iso': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu_iso': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_mu_iso_anti_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu_anti_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu_iso_anti_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_b_mu_iso_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadhad_cr_mu_iso_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_signal': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_signal': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_b': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_b': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_b_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_b_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_b_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_b_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_w': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_w': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_w_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_w_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_w_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_w_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_qcd': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_qcd': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_qcd_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_qcd_ztag_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadel_cr_qcd_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                'cutflow_hadmu_cr_qcd_dphi_inv': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
            })
        if self._plotopt==0:
            self._accumulator.add(processor.dict_accumulator({
                'met_nn_kin': hist.Hist(
                    'Events',
                    hist.Cat('dataset', 'Dataset'),
                    hist.Cat('systematic', 'Systematic'),
                    hist.Cat('region', 'Region'),
                    met_pt_bin, massreg_bin, nn_disc_bin, h_pt_bin, 
                )}
            ))
        elif self._plotopt==3:
            self._accumulator.add(processor.dict_accumulator({
                'met_nn_kin': hist.Hist(
                    'Events',
                    hist.Cat('dataset', 'Dataset'),
                    hist.Cat('systematic', 'Systematic'),
                    hist.Cat('region', 'Region'),
                    met_pup_pt_bin, met_trigger_bin,
                )}
            ))
        elif self._plotopt>0:
            self._accumulator.add(processor.dict_accumulator({
                '%s_kin'%n: hist.Hist(
                    'Events',
                    hist.Cat('dataset', 'Dataset'),
                    hist.Cat('systematic', 'Systematic'),
                    hist.Cat('region', 'Region'),
                    self._altplots[n], nn_disc_bin,
                    ) for n in self._altplots
                }
            ))


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection(dtype="uint64")
        nevents = len(events)
        weights = Weights(nevents, storeIndividual=True)
        output = self.accumulator.identity()
        
        if not isRealData:
            output['sumw'][dataset] = ak.sum(events.genWeight)
            
        # trigger
        triggermasks = {}
        for channel in ["e","mu",'had','met']:
            good_trigs = 0
            #if isRealData:
            trigger = np.zeros(nevents, dtype='bool')
            for t in self._triggers[channel]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
                    good_trigs += 1
            #selection.add('trigger'+channel, trigger)
            #del trigger
            if good_trigs < 1:
                raise ValueError("none of the following triggers found in dataset:", self._triggers[channel])
            #else:
                #selection.add('trigger'+channel, np.ones(nevents, dtype='bool'))
                #trigger = np.ones(nevents, dtype=np.bool)

            triggermasks[channel] = trigger
    
        if isRealData:
            overlap_removal = isOverlap(events, dataset, self._triggers['e'] + self._triggers['mu'] + self._triggers['had'] + self._triggers['met'], self._year)
        else:
            overlap_removal = np.ones(nevents, dtype=np.bool)

        met_filters = np.ones(nevents, dtype=np.bool)
        for t in self._metFilters:
            if not isRealData and t in ['eeBadScFilter']:
                continue
            met_filters = met_filters & events.Flag[t]

        selection.add('met_filters', met_filters)

        selection.add('met_trigger',    triggermasks['met'] & overlap_removal & met_filters)
        selection.add('hadel_trigger',  triggermasks['e']   & overlap_removal & met_filters)
        selection.add('hadmu_trigger',  triggermasks['mu']  & overlap_removal & met_filters)
        selection.add('hadhad_trigger', triggermasks['had'] & overlap_removal & met_filters)

        #jets
        if hasattr(events, 'FatJet'):
            fatjets = events.FatJet
        else:
            fatjets = events.CustomAK8Puppi

        import cachetools
        jec_cache = cachetools.Cache(np.inf)
        nojer = "NOJER" if self._skipJER else ""
        fatjets = fatjet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache)
        jets = jet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})

        shifts = [
            ({"Jet": jets, "FatJet": fatjets, "MET": met}, None),
            ({"Jet": jets.JES_jes.up, "FatJet": fatjets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
            ({"Jet": jets.JES_jes.down, "FatJet": fatjets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
        ]
        if not self._skipJER:
            shifts.extend([
                ({"Jet": jets.JER.up, "FatJet": fatjets.JER.up, "MET": met.JER.up}, "JERUp"),
                ({"Jet": jets.JER.down, "FatJet": fatjets.JER.down, "MET": met.JER.down}, "JERDown"),
            ])

        #fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['rho'] = 2*np.log(fatjets.msoftdrop/fatjets.pt) #for some reason this doesnt set the attribute properly, so you need to refer to it by the string

        ak8jets = fatjets[
            (fatjets.pt > 200)
            & (np.abs(fatjets.eta) < 2.5)
        ]

        ak8jets_p4 = ak.zip(
            {
                'pt' : ak8jets.pt,
                'eta' : ak8jets.eta,
                'phi' : ak8jets.phi,
                'mass' : ak8jets.mass
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )

        met_nopup = events.MET 
        met_nopup_p4 = ak.zip(
            {
                'pt' : met_nopup.pt,
                'eta': 0,
                'phi': met_nopup.phi,
                'mass' : 0
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )

        met = events.PuppiMET 
        met_p4 = ak.zip(
            {
                'pt' : met.pt,
                'eta': 0,
                'phi': met.phi,
                'mass' : 0
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )

        ak8_met_pair = ak.cartesian( (ak8jets_p4, met_p4) )

        ak8s = ak8_met_pair[:,:,'0']
        mets = ak8_met_pair[:,:,'1']
        ak8_met_dphi = np.abs(ak8s.delta_phi(mets))

        best_ak8_idx = ak.argmin(ak8_met_dphi, axis=1, keepdims=True)
        #indexing ak8jets like ak8s works because there is only one met per event
        best_ak8 = ak.firsts(ak.to_regular(ak8jets[best_ak8_idx]))
        best_ak8_p4 = ak.firsts(ak8jets_p4[best_ak8_idx])
        best_ak8_met_dphi = ak.firsts(ak.to_regular(ak8_met_dphi[best_ak8_idx]))

        try: #FIXME: hack to not care about PFNANO vs NN postprocessed
            nn_disc_hadel = events.IN.hadel_v6
            nn_disc_hadmu = events.IN.hadmu_v6
            nn_disc_hadhad = events.IN.hadhad_v6_multi_Higgs
            nn_disc_hadhad_qcd = events.IN.hadhad_v6_multi_QCD
            nn_disc_hadhad_wjets = events.IN.hadhad_v6_multi_WJets
    
            massreg_hadel = events.MassReg.hadel_mass
            massreg_hadmu = events.MassReg.hadmu_mass
            massreg_hadhad = events.MassReg.hadhad_mass
    
            ptreg_hadel = events.MassReg.hadel_pt
            ptreg_hadmu = events.MassReg.hadmu_pt
            ptreg_hadhad = events.MassReg.hadhad_pt
    
            ztagger_el = events.Ztagger.v6_Zee_Zhe
            ztagger_mu = events.Ztagger.hadmu_v6_multi_Zhm
            ztagger_mu_mm = events.Ztagger.hadmu_v6_multi_Zmm
            ztagger_mu_qcd = events.Ztagger.hadmu_v6_multi_QCD

        except:
            nn_disc_hadel = np.ones(nevents)
            nn_disc_hadmu = np.ones(nevents)
            nn_disc_hadhad = np.ones(nevents)
            nn_disc_hadhad_qcd = np.zeros(nevents)
            nn_disc_hadhad_wjets = np.zeros(nevents)
    
            massreg_hadel = np.ones(nevents)*125.
            massreg_hadmu = np.ones(nevents)*125.
            massreg_hadhad = np.ones(nevents)*125.
    
            ptreg_hadel = np.ones(nevents)*400.
            ptreg_hadmu = np.ones(nevents)*400.
            ptreg_hadhad = np.ones(nevents)*400.
    
            ztagger_el = np.ones(nevents)
            ztagger_mu = np.ones(nevents)
            ztagger_mu_mm = np.zeros(nevents)
            ztagger_mu_qcd = np.zeros(nevents)

        selection.add('ptreg_hadel', ptreg_hadel>280.0)
        selection.add('ptreg_hadmu', ptreg_hadmu>280.0)
        selection.add('ptreg_hadhad', ptreg_hadhad>280.0)

        selection.add('nn_disc_hadel', nn_disc_hadel>0.95)
        selection.add('nn_disc_hadmu', nn_disc_hadmu>0.95)
        selection.add('nn_disc_hadhad', nn_disc_hadhad>0.999999)

        selection.add('ztagger_el', ztagger_el < 0.01)
        selection.add('ztagger_mu', ztagger_mu > 0.95)
        selection.add('ztagger_elInv', ztagger_el >= 0.01)
        selection.add('ztagger_muInv', ztagger_mu <= 0.95)

        selection.add('jetacceptance', (
            (best_ak8.pt > 200.)
            & (np.abs(best_ak8.eta) < 2.4)
            & (best_ak8['rho'] > -6.0)
            & (best_ak8['rho'] < -1.40)
        ))

        selection.add('jetacceptance450Inv', (
            (best_ak8.pt <= 450)
            & (np.abs(best_ak8.eta) < 2.4)
            & (best_ak8['rho'] > -6.0)
            & (best_ak8['rho'] < -1.40)
        ))

        selection.add('jetacceptance400', (
            (best_ak8.pt > 400)
            & (np.abs(best_ak8.eta) < 2.4)
            & (best_ak8['rho'] > -6.0)
            & (best_ak8['rho'] < -2.0)
        ))

        selection.add('jetacceptance450', (
            (best_ak8.pt > 450)
            & (np.abs(best_ak8.eta) < 2.4)
            & (best_ak8['rho'] > -6.0)
            & (best_ak8['rho'] < -2.1)
        ))

        selection.add('jet_msd', (best_ak8.msoftdrop > 20.))
        selection.add("jetid", best_ak8.isTight)

        alljets = events.Jet[
            (events.Jet.pt>30.)
        ]

        hem_cleaning = (
            ak.any((
                (alljets.eta > -3.)
                & (alljets.eta < -1.3)
                & (alljets.phi > -1.57)
                & (alljets.phi < -0.87)
            ), -1)
            | (
                (met.phi > -1.62)
                & (met.pt < 470.)
                & (met.phi < -0.62)
            )
        )

        selection.add('hem_cleaning', ~hem_cleaning)

        ak4jets = events.Jet[
            (events.Jet.pt > 30.)
            & np.abs((events.Jet.eta) < 2.4)
            & (events.Jet.isTight) 
        ]

        #only first 5 jets for some reason
        ak4jets = ak4jets[:, :5]

        ak4jets_p4 = ak.zip(
            {
                'pt' : ak4jets.pt,
                'eta' : ak4jets.eta,
                'phi' : ak4jets.phi,
                'mass' : ak4jets.mass
            },
            behavior = vector.behavior,
            with_name = 'PtEtaPhiMLorentzVector'
        )
        
        ak4_ak8_pair = ak.cartesian( (ak4jets_p4, best_ak8_p4) )
        ak4s = ak4_ak8_pair[:,:,'0']
        ak8s = ak4_ak8_pair[:,:,'1']

        ak4_ak8_dr = ak4s.delta_r(ak8s)
        #again, indexing like this works because there is at most 1 best_ak8
        ak4_away = ak4jets[ak4_ak8_dr > 0.8]

        selection.add('antiak4btagMediumOppHem', 
                ak.max(ak4_away.btagDeepB, 1) <= self._btagWPs['medium'])
        selection.add('ak4btagMedium08', 
                ak.max(ak4_away.btagDeepB, 1) > self._btagWPs['medium'])

        ak4_met_pair = ak.cartesian( (ak4jets, met) )
        ak4s = ak4_met_pair[:,:,'0']
        mets = ak4_met_pair[:,:,'1']

        ak4_met_dphi = np.abs(ak4s.delta_phi(mets))
        minidx = ak.argmin(ak4_met_dphi, axis=1, keepdims=True)
        ak4_met_mindphi = ak4_met_dphi[minidx]
        selection.add('jetmet_dphi', best_ak8_met_dphi < np.pi/2)
        selection.add('jetmet_dphiInv', best_ak8_met_dphi >= np.pi/2)

        selection.add('met', met.pt > 20.)
        selection.add('met50', met.pt > 50.)
        selection.add('met100', met.pt > 100.)
        selection.add('methard', met.pt>150.)

        #muons
        goodmuon = (
            (events.Muon.pt > 25)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.5)
            & (np.abs(events.Muon.dxy) < 0.2)
            & events.Muon.mediumId
        )
        ngoodmuons = ak.sum(goodmuon, 1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        muons = (
            (events.Muon.pt > 10)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.5)
            & (np.abs(events.Muon.dxy) < 0.2)
            & events.Muon.looseId
        )
        nmuons = ak.sum(muons ,1)

        #electrons
        goodelectrons = (
            (events.Electron.pt > 25)
            & (np.abs(events.Electron.eta) < 2.4)
            & ((1.44 < np.abs(events.Electron.eta)) | (np.abs(events.Electron.eta) > 1.57))
            & (events.Electron.mvaFall17V2noIso_WP90)
        )
        ngoodelectrons = ak.sum(goodelectrons, 1)
        leadingelectron = ak.firsts(events.Electron[goodelectrons])

        electrons = (
            (events.Electron.pt > 10)
            & ((1.44 < np.abs(events.Electron.eta)) | (np.abs(events.Electron.eta) > 1.57))
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nelectrons = ak.sum(electrons, 1)

        #taus
        try:
            tau_coll = events.boostedTau
        except:
            tau_coll = events.Tau
        tauAntiEleId = tau_coll.idAntiEle2018

        loosetaus_el = (
            (tau_coll.pt > 20)
            & (np.abs(tau_coll.eta) < 2.3)
            & (tauAntiEleId >= 2)
        )
        goodtaus_el = (
            loosetaus_el
            & (tauAntiEleId >= 4)
        )
        goodtaus_mu = (
            (tau_coll.pt > 20)
            & (np.abs(tau_coll.eta) < 2.3)
            & (tau_coll.idAntiMu >= 1)
        )
        
        etaus_p4 = ak.zip(
            {
                'pt' : tau_coll[goodtaus_el].pt,
                'eta' : tau_coll[goodtaus_el].eta,
                'phi' : tau_coll[goodtaus_el].phi,
                'mass' : tau_coll[goodtaus_el].mass,
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        etausloose_p4 = ak.zip(
            {
                'pt' : tau_coll[loosetaus_el].pt,
                'eta' : tau_coll[loosetaus_el].eta,
                'phi' : tau_coll[loosetaus_el].phi,
                'mass' : tau_coll[loosetaus_el].mass,
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        mtaus_p4 = ak.zip(
            {
                'pt' : tau_coll[goodtaus_mu].pt,
                'eta' : tau_coll[goodtaus_mu].eta,
                'phi' : tau_coll[goodtaus_mu].phi,
                'mass' : tau_coll[goodtaus_mu].mass,
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        
        etau_ak8_pair = ak.cartesian((etaus_p4, best_ak8_p4))
        etau_ak8_dr = etau_ak8_pair[:,:,'0'].delta_r(etau_ak8_pair[:,:,'1'])

        etauloose_ak8_pair = ak.cartesian((etausloose_p4, best_ak8_p4))
        etauloose_ak8_dr = etauloose_ak8_pair[:,:,'0'].delta_r(etauloose_ak8_pair[:,:,'1'])

        mtau_ak8_pair = ak.cartesian((mtaus_p4, best_ak8_p4))
        mtau_ak8_dr = mtau_ak8_pair[:,:,'0'].delta_r(mtau_ak8_pair[:,:,'1'])

        selection.add('antiElId', ak.any(etau_ak8_dr < 0.8, -1))
        selection.add('antiMuId', ak.any(mtau_ak8_dr < 0.8, -1))
        selection.add('antiId', ak.any(etau_ak8_dr < 0.8, -1) & ak.any(mtau_ak8_dr < 0.8, -1))
        selection.add('antiIdInv',~(ak.any(etau_ak8_dr < 0.8, -1) & ak.any(mtau_ak8_dr < 0.8, -1)))

        one_el = (
            ~ak.any(electrons & ~goodelectrons, 1)
            & (nmuons == 0)
            & (ngoodmuons == 0)
            & (ngoodelectrons == 1)
        )

        one_mu = (
            ~ak.any(muons & ~goodmuon, 1)
            & (nelectrons ==0)
            & (ngoodelectrons==0)
            & (ngoodmuons == 1)
        )

        mu_p4 = ak.zip(
            {
                'pt' : leadingmuon.pt,
                'eta' : leadingmuon.eta,
                'phi' : leadingmuon.phi,
                'mass' : leadingmuon.mass
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        ele_p4 = ak.zip(
            {
                'pt' : leadingelectron.pt,
                'eta' : leadingelectron.eta,
                'phi' : leadingelectron.phi,
                'mass' : leadingelectron.mass
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        nolep_p4 = ak.zip(
            {
                'pt' : ak.zeros_like(leadingelectron.pt),
                'eta' : ak.zeros_like(leadingelectron.pt),
                'phi' : ak.zeros_like(leadingelectron.pt),
                'mass' : ak.zeros_like(leadingelectron.pt),
            },
            behavior = vector.behavior,
            with_name = 'PtEtaPhiMLorentzVector'
        )

        leadinglep = ak.where(one_el, ele_p4,
                ak.where(one_mu, mu_p4, nolep_p4))

        mu_miso = leadingmuon.miniPFRelIso_all
        el_miso = leadingelectron.miniPFRelIso_all
        leadinglepmiso = ak.where(one_el, el_miso,
                ak.where(one_mu, mu_miso, 0))

        mt_lepmet = np.sqrt(2.*leadinglep.pt*met_p4.pt*(1-np.cos(met_p4.delta_phi(leadinglep))))
        mt_jetmet = np.sqrt(2.*best_ak8_p4.pt*met_p4.pt*(1-np.cos(met_p4.delta_phi(best_ak8_p4))))

        selection.add('mt_lepmet', (mt_lepmet < 60.))
        selection.add('mt_lepmetInv', (mt_lepmet>=60.))

        selection.add('noleptons', ( 
            (nmuons==0)
            & (nelectrons==0)
            & (ngoodmuons ==0)
            & (ngoodelectrons==0)
        ))

        selection.add('onemuon', one_mu)
        selection.add('oneelec', one_el)

        selection.add('muonkin', (
            (leadingmuon.pt > 30.)
            & (np.abs(leadingmuon.eta) < 2.4)
        ))
        selection.add('eleckin', (
            (leadingelectron.pt > 40.)
            & (np.abs(leadingelectron.eta) < 2.4)
        ))

        muon_ak8_dphi = np.abs(ak.fill_none(best_ak8_p4.delta_phi(mu_p4), 0))
        ele_ak8_dphi = np.abs(ak.fill_none(best_ak8_p4.delta_phi(ele_p4), 0)) 
        selection.add('muonDphiAK8', muon_ak8_dphi > 2*np.pi/3)
        selection.add('elecDphiAK8', ele_ak8_dphi > 2*np.pi/3)

        lep_ak8_dr = best_ak8_p4.delta_r(leadinglep)
        selection.add('lepDrAK8', lep_ak8_dr < 0.8)
        selection.add('lepDrAK8Inv', lep_ak8_dr >= 0.8)

        selection.add('muonIso', (
            ((leadingmuon.pt > 30)
            & (leadingmuon.pt < 55)
            & (leadingmuon.pfRelIso04_all < 0.25)
            ) 
            | ((leadingmuon.pt >= 55 )
            & (leadingmuon.miniPFRelIso_all < 0.05))
        ))

        selection.add('muonIsoInv', (
            ((leadingmuon.pt > 30)
            & (leadingmuon.pt < 55)
            & (leadingmuon.pfRelIso04_all >= 0.25)
            ) 
            | ((leadingmuon.pt >= 55) 
            & (leadingmuon.miniPFRelIso_all >= 0.05))
        ))

        selection.add('elecIso', (
            ((leadingelectron.pt > 30 )
            & (leadingelectron.pt < 120)
            & (leadingelectron.pfRelIso03_all < 0.1))
            | ((leadingelectron.pt >= 120)
            & (leadingelectron.miniPFRelIso_all < 0.05))
        ))

        selection.add('elecIsoInv', (
            ((leadingelectron.pt > 30 )
            & (leadingelectron.pt < 120)
            & (leadingelectron.pfRelIso03_all >= 0.1))
            | ((leadingelectron.pt >= 120)
            & (leadingelectron.miniPFRelIso_all >= 0.05))
        ))

        jet_lep_p4 = best_ak8_p4 - leadinglep

        '''
        apply scale factors/weights/something
        I left this as untouched as I could without getting errors
        I make no claims that this does what it's suppossed to do
        This is because I don't really know what it's supposed to do
        See also corrections.py and common.py
        '''
        if isRealData:
            genflavor = ak.zeros_like(best_ak8.pt)
            w_hadhad = deepcopy(weights)
            w_hadhadmet = deepcopy(weights)
            w_hadel = deepcopy(weights)
            w_hadmu = deepcopy(weights)
            genHTauTauDecay = ak.zeros_like(best_ak8.pt)
            genHadTau1Decay = ak.zeros_like(best_ak8.pt)
            genHadTau2Decay = ak.zeros_like(best_ak8.pt)
            genHadTau2Decay = ak.zeros_like(best_ak8.pt)
            genTauTaudecay = ak.zeros_like(best_ak8.pt)
        else:
            weights.add('genweight', events.genWeight)
            #weights.add('L1PreFiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Down)
            add_pileup_weight(weights, events.Pileup.nPU, self._year)
            if "LHEPdfWeight" in events.fields:
                add_pdf_weight(weights, events.LHEPdfWeight)
            else:
                add_pdf_weight(weights, None)
            if "LHEScaleWeight" in events.fields:
                add_scalevar_7pt(weights, events.LHEScaleWeight)
                add_scalevar_3pt(weights, events.LHEScaleWeight)
            else:
                add_scalevar_7pt(weights,[])
                add_scalevar_3pt(weights,[])
            bosons = getBosons(events)
            genBosonPt = ak.fill_none(ak.pad_none(bosons.pt, 1, clip=True), 0)
            #I don't have an implementation of these
            add_TopPtReweighting(weights, ak.pad_none(getParticles(events,6,6,['isLastCopy']).pt, 2, clip=True), self._year, dataset) #123 gives a weight of 1
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            genflavor = matchedBosonFlavor(best_ak8, bosons)
            genlepflavor = matchedBosonFlavorLep(best_ak8, bosons)
            genHTauTauDecay, genHadTau1Decay, genHadTau2Decay = getHTauTauDecayInfo(events,True)
            w_hadhad = deepcopy(weights)
            w_hadhadmet = deepcopy(weights)
            w_hadel = deepcopy(weights)
            w_hadmu = deepcopy(weights)
            #also need implementation here
            add_LeptonSFs(w_hadel, leadinglep, self._year, "elec")
            add_LeptonSFs(w_hadmu, leadinglep, self._year, "muon")
            add_METSFs(w_hadhadmet, met.pt, self._year+self._yearmod)
            #add_TriggerWeight(w_hadhad, best_ak8.msoftdrop, best_ak8.pt, leadinglep.pt, self._year, "hadhad")

        regions = {
            'hadhad_signal':               ['hadhad_trigger', 'noleptons', 'jetacceptance450', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met','antiId','jetmet_dphi'],
            'hadhad_signal_met':           ['met_trigger', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem','antiId','jetmet_dphi'],
            'hadhad_cr_anti_inv':          ['met_trigger', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem','antiIdInv','jetmet_dphi'],
            'hadhad_cr_dphi_inv':          ['met_trigger', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem','antiId','jetmet_dphiInv'],
            #'hadhad_cr_b':                 ['hadhad_trigger', 'noleptons', 'jetacceptance450', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met'],
            'hadhad_cr_b_met':             ['met_trigger', 'methard', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08','antiId','jetmet_dphi'],
            'hadhad_cr_b_met_anti_inv':    ['met_trigger', 'methard', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08','antiIdInv','jetmet_dphi'],
            'hadhad_cr_b_met_dphi_inv':    ['met_trigger', 'methard', 'noleptons', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08','antiId','jetmet_dphiInv'],
            #'hadhad_cr_b_mu':              ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8Inv', 'muonIsoInv'],
            'hadhad_cr_b_mu_iso':          ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8Inv', 'muonIso','antiId','jetmet_dphi'],
            'hadhad_cr_mu':                ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIsoInv','antiId','jetmet_dphi'],
            'hadhad_cr_mu_iso':            ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIso','antiId','jetmet_dphi'],
            'hadhad_cr_b_mu_iso_anti_inv': ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8Inv', 'muonIso','antiIdInv','jetmet_dphi'],
            'hadhad_cr_mu_anti_inv':       ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIsoInv','antiIdInv','jetmet_dphi'],
            'hadhad_cr_mu_iso_anti_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIso','antiIdInv','jetmet_dphi'],
            'hadhad_cr_b_mu_iso_dphi_inv': ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8Inv', 'muonIso','antiId','jetmet_dphiInv'],
            'hadhad_cr_mu_dphi_inv':       ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIsoInv','antiId','jetmet_dphiInv'],
            'hadhad_cr_mu_iso_dphi_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8Inv', 'muonIso','antiId','jetmet_dphiInv'],
            'hadmu_signal':          ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'muonIso', 'jetmet_dphi','ztagger_mu'],
            'hadel_signal':          ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'elecIso', 'jetmet_dphi','ztagger_el'],
            'hadmu_cr_ztag_inv':     ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'muonIso', 'jetmet_dphi','ztagger_muInv'],
            'hadel_cr_ztag_inv':     ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'elecIso', 'jetmet_dphi','ztagger_elInv'],
            'hadmu_cr_dphi_inv':     ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'muonIso', 'jetmet_dphiInv','ztagger_mu'],
            'hadel_cr_dphi_inv':     ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmet', 'elecIso', 'jetmet_dphiInv','ztagger_el'],
            'hadmu_cr_qcd_ztag_inv': ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'muonIsoInv','jetmet_dphi','ztagger_muInv'],
            'hadel_cr_qcd_ztag_inv': ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'elecIsoInv','jetmet_dphi','ztagger_elInv'],
            'hadmu_cr_qcd_dphi_inv': ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'muonIsoInv','jetmet_dphiInv','ztagger_mu'],
            'hadel_cr_qcd_dphi_inv': ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'elecIsoInv','jetmet_dphiInv','ztagger_el'],
            'hadmu_cr_qcd':          ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'muonIsoInv','jetmet_dphi','ztagger_mu'],
            'hadel_cr_qcd':          ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'elecIsoInv','jetmet_dphi','ztagger_el'],
            'hadmu_cr_b':            ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'muonIso','jetmet_dphi','ztagger_mu'],
            'hadel_cr_b':            ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'elecIso','jetmet_dphi','ztagger_el'],
            'hadmu_cr_b_ztag_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'muonIso','jetmet_dphi','ztagger_muInv'],
            'hadel_cr_b_ztag_inv':   ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'elecIso','jetmet_dphi','ztagger_elInv'],
            'hadmu_cr_b_dphi_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'muonIso','jetmet_dphiInv','ztagger_mu'],
            'hadel_cr_b_dphi_inv':   ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'ak4btagMedium08', 'met', 'lepDrAK8', 'elecIso','jetmet_dphiInv','ztagger_el'],
            'hadmu_cr_w':            ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'muonIso', 'jetmet_dphi','ztagger_mu'],
            'hadel_cr_w':            ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'elecIso', 'jetmet_dphi','ztagger_el'],
            'hadmu_cr_w_ztag_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'muonIso', 'jetmet_dphi','ztagger_muInv'],
            'hadel_cr_w_ztag_inv':   ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'elecIso', 'jetmet_dphi','ztagger_elInv'],
            'hadmu_cr_w_dphi_inv':   ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'muonIso', 'jetmet_dphiInv','ztagger_mu'],
            'hadel_cr_w_dphi_inv':   ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'antiak4btagMediumOppHem', 'met', 'lepDrAK8', 'mt_lepmetInv', 'elecIso', 'jetmet_dphiInv','ztagger_el'],
            #'noselection': [],
        }
        if self._plotopt==3:
            regions = {
                'hadmu_base':          ['onemuon', 'muonkin', 'muonIso'],
                'hadel_base':          ['oneelec', 'eleckin', 'elecIso'],
                'hadmu_lep':          ['hadmu_trigger', 'onemuon', 'muonkin', 'muonIso'],
                'hadel_lep':          ['hadel_trigger', 'oneelec', 'eleckin', 'elecIso'],
                'hadmu_jet':          ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'muonIso'],
                'hadel_jet':          ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'elecIso'],
                'hadmu_signal':          ['hadmu_trigger', 'onemuon', 'muonkin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'mt_lepmet', 'muonIso'],
                'hadel_signal':          ['hadel_trigger', 'oneelec', 'eleckin', 'jetacceptance', 'jet_msd', 'jetid', 'lepDrAK8', 'mt_lepmet', 'elecIso'],
            }
            if isRealData:
                regions['hadmu_base'].insert(0,'hadmu_trigger')
                regions['hadel_base'].insert(0,'hadel_trigger')

        if (self._year == '2018'):
            for r in regions:
                regions[r].append('hem_cleaning')

        if self._plotopt>0 and self._plotopt!=3:
            for r in regions:
                if 'hadhad' in r:
                    regions[r].extend(['ptreg_hadhad','methard'])
                elif 'hadel' in r:
                    regions[r].extend(['ptreg_hadel','met100'])
                elif 'hadmu' in r:
                    regions[r].extend(['ptreg_hadmu','met100'])

        w_dict = {}
        for r in regions:
            if 'hadhad' in r and 'met' not in r:
                w_dict[r] = w_hadhad 
            elif 'hadhad' in r and 'met' in r:
                w_dict[r] = w_hadhadmet 
            if 'hadel' in r:
                w_dict[r] = w_hadel
            if 'hadmu' in r:
                w_dict[r] = w_hadmu

        for r in regions:
            allcuts_reg = set()
            output['cutflow_%s'%r][dataset]['none'] += float(w_dict[r].weight().sum())
            if self._plotopt==0:
                if 'hadhad' in r:
                    addcuts = ['methard' if 'met' in r else 'met50','ptreg_hadel','nn_disc_hadhad'] 
                elif 'hadel' in r:
                    addcuts = ['met100','ptreg_hadel','nn_disc_hadel'] 
                elif 'hadmu' in r:
                    addcuts = ['met100','ptreg_hadmu','nn_disc_hadmu']
            elif self._plotopt==3:
                addcuts = []
            else:
                if 'hadhad' in r:
                    addcuts = ['nn_disc_hadhad'] 
                elif 'hadel' in r:
                    addcuts = ['met100','ptreg_hadel','nn_disc_hadel'] 
                elif 'hadmu' in r:
                    addcuts = ['met100','ptreg_hadmu','nn_disc_hadmu']
            for cut in regions[r]+addcuts:
                allcuts_reg.add(cut)
                output['cutflow_%s'%r][dataset][cut] += float(w_dict[r].weight()[selection.all(*allcuts_reg)].sum())

        systematics = [
            None,
            #'jet_triggerUp',
            #'jet_triggerDown',
            #'btagWeightUp',
            #'btagWeightDown',
            #'btagEffStatUp',
            #'btagEffStatDown',
            #'TopPtReweightUp',
            #'TopPtReweightDown',
            #'L1PreFiringUp',
            #'L1PreFiringDown',
        ]
        if isRealData:
            systematics = [None]

        def fill(region, systematic, wmod=None, realData=False, addcut=None, addname=""):
            selections = regions[region]
            cut = selection.all(*selections)
            if addcut is not None:
                cut = cut * addcut
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if not realData:
                    weight = w_dict[region].weight(modifier=systematic)
                else:
                    weight = w_dict[region].weight(modifier=None)
            else:
                weight = w_dict[region].weight() * wmod

            #weight = zero if cut fails
            #seems reasonable enough, right?
            weight = ak.where(cut, weight, 0)

            '''
            I /think/ this is equivalent to what was here before...
            '''
            if 'hadhad' in region:
                nn_disc = nn_disc_hadhad
                massreg = massreg_hadhad
                ptreg = ptreg_hadhad
                antilep = ak.any(mtau_ak8_dr < 0.8, -1) & ak.any(etauloose_ak8_dr < 0.8, -1)
                ztagger = ztagger_el
            if 'hadel' in region:
                nn_disc = nn_disc_hadel
                massreg = massreg_hadel
                ptreg = ptreg_hadel
                antilep = ak.any(etau_ak8_dr < 0.8, -1)
                antilep = ak.any(etauloose_ak8_dr < 0.8, -1) + antilep - 1
                ztagger = ztagger_el
            if 'hadmu' in region:
                nn_disc = nn_disc_hadmu
                massreg = massreg_hadmu
                ptreg = ptreg_hadmu
                antilep = ak.any(mtau_ak8_dr < 0.8, -1)
                ztagger = ztagger_mu

            bmaxind = ak.argmax(ak4_away.btagDeepB, -1)

            if self._plotopt==0:
                output['met_nn_kin'].fill(
                    dataset=dataset+addname,
                    region=region,
                    systematic=sname,
                    met_pt=normalize(met_p4.pt),
                    massreg=normalize(massreg),
                    nn_disc=normalize(nn_disc),
                    h_pt=normalize(ptreg),
                    weight=weight,
                )
            elif self._plotopt==3:
                output['met_nn_kin'].fill(
                    dataset=dataset+addname,
                    region=region,
                    systematic=sname,
                    met_pup_pt=normalize(met_p4.pt),
                    met_trigger=normalize(triggermasks['met']),
                    weight=weight,
                )
            elif self._plotopt>0:
                altvars = {
                    'jet_pt':best_ak8.pt, 
                    'jet_eta':best_ak8.eta, 
                    'jet_msd':best_ak8.msoftdrop, 
                    'mt_lepmet':mt_lepmet, 
                    'mt_jetmet':massreg, 
                    'lep_pt':leadinglep.pt, 
                    'lep_eta':leadinglep.eta, 
                    'lep_jet_dr':lep_ak8_dr, 
                    'n2b1':best_ak8.n2b1, 
                    'jetlep_m':jet_lep_p4.mass, 
                    'met_nopup_pt':met_nopup_p4.pt, 
                    'met_pup_pt':met_p4.pt, 
                    'jetmet_dphi':best_ak8_met_dphi, 
                    'ak4met_dphi':ak4_met_mindphi, 
                    'massreg':massreg, 
                    'ztagger':ztagger, 
                    'lep_miso':leadinglepmiso, 
                    'nn_hadhad':nn_disc_hadhad , 
                    'nn_hadhad_qcd':nn_disc_hadhad_qcd, 
                    'nn_hadhad_wjets':nn_disc_hadhad_wjets, 
                    'ztagger_mu_qcd':ztagger_mu_qcd, 
                    'ztagger_mu_mm':ztagger_mu_mm, 
                    'ztagger_mu_hm':ztagger_mu, 
                    'nn_hadel':nn_disc_hadel, 
                    'nn_hadmu':nn_disc_hadmu, 
                    'antilep':antilep
                }
                for var in self._altplots:
                    output['%s_kin'%var].fill(
                        **{'dataset':dataset+addname,
                        'region':region,
                        'systematic':sname,
                        'weight':weight,
                        **{var:normalize(altvars[var]), 'nn_disc':normalize(nn_disc)}}
                    )

        for region in regions:
            for systematic in systematics:
                if 'DYJets' in dataset:
                    fill(region, systematic, realData=isRealData, addcut=ak.to_numpy(ak.any(genlepflavor>=2, -1)).flatten(), addname="_Zem")
                    fill(region, systematic, realData=isRealData, addcut=ak.to_numpy(ak.any(genlepflavor==1, -1)).flatten(), addname="_Ztt")
                else:
                    fill(region, systematic, realData=isRealData)

        return output

    def postprocess(self, accumulator):
        return accumulator
