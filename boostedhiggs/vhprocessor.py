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
np.seterr(invalid="ignore")


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


class vhProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017", #note that this has been hardcoded!! need to be careful of the year or change to make it adapt/year
        yearmod="",
        #channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        apply_trigger=True,
    ):

        self._year = year
        self._yearmod = yearmod
        #self._channels = channels
        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]
                print('self.HLTs', self._HLTs)

        # apply trigger in selection?
        self.apply_trigger = apply_trigger

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        # b-tagging corrector
        self._btagWPs = btagWPs["deepJet"][year + yearmod]
        # self._btagSF = BTagCorrector("M", "deepJet", year, yearmod)

        if year == "2018":  #TO DO _ CHANGE PER dataset - try first using Cristina's dataset and no channels
            #self.dataset_per_ch = {
            self.dataset = {
	        "DoubleEG",
		"DoubleMu",
		"MuonEG"
            }
        else: #fix this to add in single muon and electron and also one is missing in 2018
            self.dataset = {
	        "SingleElectron",
		"SingleMuon", #use cristina's - maybe code is failing b/c of lack of weights?
                "DoubleMu"
            }

        # do inference
        self.inference = inference
        # for tagger model and preprocessing dict
        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

        self.common_weights = ["genweight", "L1Prefiring", "pileup"]

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        #channels = channel if channel else self._channels
        #for ch in channels:
        self.selections.add(name, sel)
        selection = self.selections.all(*self.selections.names)
        print('selection', selection)
        if self.isMC:
            #weight = self.weights.partial_weight(self.weights + self.common_weights)
            #weight = self.weights.partial_weight(self.weights_vh + self.common_weights)
            weight = self.weights.partial_weight(self.common_weights)
            self.cutflows[name] = float(weight[selection].sum())
        else:
            self.cutflows[name] = np.sum(selection)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        print('got an event')
        dataset = events.metadata["dataset"]
        nevents = len(events)
        print('nevents', nevents)

        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        print('self.weights', self.weights)
        #self.weights_vh = []
        self.selections = {}
        self.cutflows = {}
        #for ch in self._channels:
        #self.weights = []  #not sure what to do here...... 
        self.selections = PackedSelection()

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

#******************TRIGGER******************************************
        # trigger            Def: self._HLTs = json.load(f)[self._year]
        trigger = {}          #to do - make a dictionary looping over triggers, put actual sel logic below
        #for ch in self._channels:
        #trigger = np.zeros(nevents, dtype="bool") #cristina said to move this below???
        #vhTriggerList = ['DoubleMuon', 'DoubleEG', 'MuonEG', 'ele', 'mu'] #for testing, not full list add also EGAmma for 2018
        vhTriggerList = ['ele', 'mu', 'DoubleMuon'] #for testing, not full list add also EGAmma for 2018
        for trig in vhTriggerList:
            print('trig', trig)
            print('self._HLTs[trig]', self._HLTs[trig])
            trigger[trig] = np.zeros(nevents,dtype="bool")
            for t in self._HLTs[trig]:
                print('t', t)
                if t in events.HLT.fields:
                    trigger[trig] = trigger[trig] | events.HLT[t]
                    print('trigger[trig]', trigger[trig])
                    print('try a few events', ak.to_list(trigger[trig])[0:3])
                    #print('length of trigger', len(trigger))
        print('trigger - is this a dictionary', trigger)
#******************TRIGGER******************************************

        # metfilters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # taus (will need to refine to avoid overlap with htt)
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        #add flavor tag to muons and electrons
        good_muons = ak.with_field(events.Muon,0, 'flavor')
        good_electrons = ak.with_field(events.Electron,1, 'flavor')

        loose_muons = (
            good_muons[  (((good_muons.pt > 30) & (good_muons.pfRelIso04_all < 0.25)) | (good_muons.pt > 55))
            & (np.abs(good_muons.eta) < 2.4)
            & (good_muons.looseId) ]
        )
        #n_loose_muons = ak.sum(loose_muons, axis=1) #note since i changed to recors to add flavor, this summing doesn't work

        good_muons = (
            good_muons[(good_muons.pt > 30)
            & (np.abs(good_muons.eta) < 2.4)
            & (np.abs(good_muons.dz) < 0.1)
            & (np.abs(good_muons.dxy) < 0.05)
            & (good_muons.sip3d <= 4.0)
            & good_muons.mediumId ]
        )
        #n_good_muons = ak.sum(good_muons, axis=1)

        loose_electrons = (
            good_electrons[(((good_electrons.pt > 38) & (good_electrons.pfRelIso03_all < 0.25)) | (good_electrons.pt > 120))
            & (np.abs(good_electrons.eta) < 2.4)
            & ((np.abs(good_electrons.eta) < 1.44) | (np.abs(good_electrons.eta) > 1.57))
            & (good_electrons.cutBased >= good_electrons.LOOSE)]
        )
        #n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            good_electrons[(good_electrons.pt > 38) 
            #good_electrons[(good_electrons.pt > 40) #for the selection later, asks for pt > 40
            & (np.abs(good_electrons.eta) < 2.4)
            & ((np.abs(good_electrons.eta) < 1.44) | (np.abs(good_electrons.eta) > 1.57))
            & (np.abs(good_electrons.dz) < 0.1)
            & (np.abs(good_electrons.dxy) < 0.05)
            & (good_electrons.sip3d <= 4.0)
            & (good_electrons.mvaFall17V2noIso_WP90)  ]
        )

        goodleptons = ak.concatenate([good_muons,good_electrons ], axis=1)  
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

#************************************************************************************************************
#add this section below - get the leptons that belong to the Z, then get the other leptons - use the highest pt lepton of the latter for the candidate lepton
#note: need to get rid of two sep. channels or combine
        minThreeLeptonsMask = ak.num(goodleptons, axis=1) >= 3
        minThreeLeptons = ak.mask(goodleptons, minThreeLeptonsMask[:,None])

        #print('minThreeLeptonsMask', minThreeLeptonsMask)
        #print('minThreeLeptons', minThreeLeptons)

        lepton_pairs = ak.argcombinations(minThreeLeptons, 2, fields=['first', 'second'])
        lepton_pairs = ak.fill_none(lepton_pairs, [], axis=0)

        OSSF_pairs = lepton_pairs[    (minThreeLeptons[lepton_pairs['first']].charge != minThreeLeptons[lepton_pairs['second']].charge) & (minThreeLeptons[lepton_pairs['first']].flavor == minThreeLeptons[lepton_pairs['second']].flavor  )  ]

        closest_pairs = OSSF_pairs[ak.local_index(OSSF_pairs) == ak.argmin(np.abs((minThreeLeptons[OSSF_pairs['first']] + minThreeLeptons[OSSF_pairs['second']]).mass - 91.2), axis=1)]
        closest_pairs = ak.fill_none(closest_pairs, [], axis=0)

        new1 = closest_pairs.first #this gives the index of the first lepton in the lepton pair that adds us best to the invariant mass of the Z
        new2 = closest_pairs.second #this gives the index of the second lepton

        #invariant Z mass
        ZLeptonMass = ((minThreeLeptons[closest_pairs.first]+minThreeLeptons[closest_pairs.second]).mass)
        desired_length = np.max(ak.num(ZLeptonMass))
        ZLepMass = ak.to_numpy(ak.fill_none(ak.pad_none(ZLeptonMass, desired_length), 0))

        print('ZLepMass', ZLepMass)
        ZLepMass = ak.ravel(ZLepMass)
        print('ZLepMass', ZLepMass)

        remainingLeptons = minThreeLeptons[ (ak.local_index(minThreeLeptons)!= ak.any(new1, axis=1))& (ak.local_index(minThreeLeptons) != ak.any(new2, axis=1))]                   
        candidatelep = ak.firsts(remainingLeptons)  # pick highest pt 

        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton
        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton
        mu_mvaId = candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)  # MVA-ID for candidate lepton


        #for now, comment this out, since we don't need specifically muon/lepton; can add one for lepton i guess, be careful can't use 'events' here 
        #mu_highPtId = ak.firsts(events.Muon[good_muons]).highPtId
        #ele_highPtId = ak.firsts(events.Electron[good_electrons]).cutBased_HEEP

        # jets
        goodjets = events.Jet[
            (events.Jet.pt > 30) & (abs(events.Jet.eta) < 5.0) & events.Jet.isTight & (events.Jet.puId > 0)
        ]
        # reject EE noisy jets for 2017
        if self._year == "2017":
            goodjets = goodjets[(goodjets.pt > 50) | (abs(goodjets.eta) < 2.65) | (abs(goodjets.eta) > 3.139)]
        ht = ak.sum(goodjets.pt, axis=1)

        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjets["qcdrho"] = 2 * np.log(fatjets.msdcorr / fatjets.pt)

        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        # for leptonic channel: first clean jets and leptons by removing overlap, then pick candidate_fj closest to the lepton
        lep_in_fj_overlap_bool = good_fatjets.delta_r(candidatelep_p4) > 0.1
        good_fatjets = good_fatjets[lep_in_fj_overlap_bool]
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # MET
        met = events.MET

        print('met', met)
        mt_lep_met = np.sqrt(
            2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        print('mt_lep_met', mt_lep_met)
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
        n_jets_outside_ak8 = ak.sum(goodjets.delta_r(candidatefj) > 0.8, axis=1)
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
        The transverse momentum of the jets in this region is typically under-measured, this results in over-measured MET. It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = (
                events.run
                >= 319077
                & ak.any(
                    (
                        (events.Jet.pt > 30.0)
                        & (events.Jet.eta > -3.2)
                        & (events.Jet.eta < -1.3)
                        & (events.Jet.phi > -1.57)
                        & (events.Jet.phi < -0.87)
                    ),
                    -1,
                )
                | ((met.phi > -1.62) & (met.pt < 470.0) & (met.phi < -0.62))
            )

        # output tuple variables
        variables = {
	#"lep": {
		"Zmass": ZLepMass, #for now put this in the electron channel, will need to separate by channel
#	},
                #"fj_pt": candidatefj.pt,
                #"fj_msoftdrop": candidatefj.msdcorr,
                #"fj_bjets_ophem": ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1),
                #"lep_pt": candidatelep.pt,
               # "lep_isolation": lep_reliso,
               # "lep_misolation": lep_miso,
               # "lep_fj_m": lep_fj_m,
               # "lep_fj_dr": lep_fj_dr,
               # "lep_met_mt": mt_lep_met,
               # "met_fj_dphi": met_fjlep_dphi,
#	"common": {
                "met": met.pt,
                "ht": ht,
#		},
                #"nfj": n_fatjets,
                #"nj": n_jets_outside_ak8,
                #"deta": deta,
                #"mjj": mjj,
        }

        # gen matching for signal
    #    if (("HToWW" or "HWW") in dataset) and self.isMC:
    #        matchHWW = match_HWW(events.GenPart, candidatefj)
    #        variables["lep"]["gen_Hpt"] = ak.firsts(matchHWW["matchedH"].pt)
    #        variables["lep"]["gen_Hnprongs"] = matchHWW["hWW_nprongs"]
    #        variables["lep"]["gen_iswlepton"] = matchHWW["iswlepton"]
    #        variables["lep"]["gen_iswstarlepton"] = matchHWW["iswstarlepton"]

        # gen matching for background
    #    if ("WJets" in dataset) or ("ZJets" in dataset) and self.isMC:
    #        matchV = match_V(events.GenPart, candidatefj)
    #        if "WJetsToLNu" in dataset:
    #            variables["lep"]["gen_isVlep"] = matchV["gen_isVlep"]
    #        if ("WJetsToQQ" in dataset) or ("ZJetsToQQ" in dataset):
    #            variables["lep"]["gen_isVqq"] = matchV["gen_isVqq"]
    #    if ("TT" in dataset) and self.isMC:
    #        matchT = match_Top(events.GenPart, candidatefj)
    #        variables["lep"]["gen_isTop"] = matchT["gen_isTopbmerged"]
    #        variables["lep"]["gen_isToplep"] = matchT["gen_isToplep"]
    #        variables["lep"]["gen_isTopqq"] = matchT["gen_isTopqq"]

        # if trigger is not applied then save the trigger variables
  #      if not self.apply_trigger:
  #          variables["lep"]["cut_trigger_iso"] = trigger_iso[ch]
  #          variables["lep"]["cut_trigger_noniso"] = trigger_noiso[ch]

        # let's save the hem veto as a cut for now
  #      if self._year == "2018":
  #          variables["common"]["hem_cleaning"] = hem_cleaning

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
        if self.isMC:
            self.weights.add("genweight", events.genWeight)
            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn
                )
            add_pileup_weight(self.weights, self._year, self._yearmod, nPU=ak.to_numpy(events.Pileup.nPU))

            add_lepton_weight(self.weights, candidatelep, self._year + self._yearmod, "muon")
            add_lepton_weight(self.weights, candidatelep, self._year + self._yearmod, "electron")
            # self._btagSF.addBtagWeight(bjets_away_lepfj, self.weights, "lep")
            add_VJets_kFactors(self.weights, events.GenPart, dataset)

            # store the final common weight
            #variables["common"]["weight"] = self.weights.partial_weight(self.common_weights)
            variables["weight"] = self.weights.partial_weight(self.common_weights)
                                       #Def above of: self.common_weights = ["genweight", "L1Prefiring", "pileup"]
            for key in self.weights._weights.keys():
                print('self.weights._weights.keys()', self.weights._weights.keys())

                # ignore btagSFlight/bc for now
                #if "btagSFlight" in key or "btagSFbc" in key:
                 #   continue
            #    if "lep" in key:
             #       varkey = "lep"
              #  else:
               #     varkey = "common"
                # store the individual weights (ONLY for now until we debug)
              #  variables[varkey][f"weight_{key}"] = self.weights.partial_weight([key])
              #  print('key', key)
                #if varkey in self.weights_vh.keys():
              #  self.weights_vh[varkey].append(key)
           # print('self.weights_vh', self.weights_vh)

            #if len(self.weights) > 0:
             #   variables[f"weight"] = self.weights.partial_weight(self.weights)
            # NOTE: to add variations:
            # for var in self.weights.variations:
            #     variables["common"][f"weight_{key}"] = self.weights.weight(key)

        """
        Selection and cutflows.
        """
        print('before any selections')
        self.add_selection("all", np.ones(nevents, dtype="bool"))

        

        if self.apply_trigger: #try first cristina's
            print('muon trigger', trigger['mu'])
            #if trigger['mu']:
             #   self.add_selection("trigger", trigger['mu'])
              #  print('added trigger: trigger for electron', trigger['mu'])
            #else:
             #   self.add_selection("trigger", trigger['ele'])
              #  print('added trigger: trigger for electron', trigger['ele'])
            self.add_selection("trigger", trigger['mu'])


        self.add_selection("metfilters", metfilters)
        self.add_selection(name="ht", sel=(ht > 200))
        self.add_selection(
            name="antibjettag",
            sel=(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1) < self._btagWPs["M"])
        )
        self.add_selection(
            name="ThreeOrMoreLeptons", sel=(minThreeLeptonsMask == True),
        )
    #    self.add_selection(name="leptonKin", sel=(candidatelep.pt > 30))
        #self.add_selection(name="leptonKin", sel=(candidatelep.pt > 40), channel=["ele"]) #no distinction b/t e and mu
   #     self.add_selection(name="fatjetKin", sel=candidatefj.pt > 200)
   #     self.add_selection(name="leptonInJet", sel=(lep_fj_dr < 0.8))

        #self.add_selection(name="notaus", sel=(n_loose_taus_mu == 0), channel=["mu"])
        #self.add_selection(name="notaus", sel=(n_loose_taus_ele == 0), channel=["ele"])

        # initialize pandas dataframe
        output = {}

        #for ch in self._channels:
        fill_output = True
            # for data, only fill output for the dataset needed
        if not self.isMC and self.dataset not in dataset:
            fill_output = False

        selection = self.selections.all(*self.selections.names)
        print('selection', selection)

            # only fill output for that channel if the selections yield any events
        if np.sum(selection) <= 0:
            fill_output = False

        if fill_output:
         #   keys = ["lep", "common"]
            out = {}
          #  for key in keys:
            #for var, item in variables[key].items():
            print('variables.items()', variables.items)
            for var, item in variables.items():
                print('var, item', var, item)
                    # pad all the variables that are not a cut with -1
                pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                        # fill out dictionary
                print('pad_item', pad_item)
                out[var] = item
                print('out[var]', out[var])
                    # fill the output dictionary after selections
            output = {key: value[selection] for (key, value) in out.items()}
            print('output', output)

            # fill inference
            if self.inference:
                print("pre-inference")
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path, events[selection], fj_idx_lep[selection]
                )
                print("post-inference")
                print(pnet_vars)

                output = {**output, **{key: value for (key, value) in pnet_vars.items()}}
        #else:
         #   output = {}
          #  print('output, line 575', output)

        # convert arrays to pandas
        print('trying pandas')
        print('output - line 578', output)
        if not isinstance(output, pd.DataFrame):
            print('trying to convert to pandas')
            output = self.ak_to_pandas(output) #ak.to_dataframe
            print('output - line 583', output)

        # now save pandas dataframes
        print('line 586')
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname
        print('fname', fname)

        #for ch in self._channels:  # creating directories for each channel
        if not os.path.exists(self._output_location):
            os.makedirs(self._output_location)
        if not os.path.exists(self._output_location + "/parquet"):
            os.makedirs(self._output_location + "/parquet")

#need a new version without channel
        self.save_dfs_parquet(fname, output)

        # return dictionary with cutflows
        return {dataset: {"mc": self.isMC, self._year: {"sumgenweight": sumgenweight, "cutflows": self.cutflows}}}

    def postprocess(self, accumulator):
        return accumulator
