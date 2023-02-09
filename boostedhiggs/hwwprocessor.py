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
from boostedhiggs.corrections import add_lepton_weight, add_pileup_weight, add_VJets_kFactors, corrected_msoftdrop
from boostedhiggs.utils import get_neutrino_z, match_H, match_Top, match_V

from .run_tagger_inference import runInferenceTriton

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


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

    def add_selection(self, name: str, sel: np.ndarray, channel: list = None):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = (
            channel
            if (channel is not None and channel in self._channels)
            else self._channels
        )
        for ch in channels:
            if ch not in self._channels:
                continue
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)
            if self.isMC:
                weight = self.weights.partial_weight(self.weights_per_ch[ch] + self.common_weights)
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata["dataset"]
        nevents = len(events)

        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.weights_per_ch = {}
        self.selections = {}
        self.cutflows = {}
        for ch in self._channels:
            self.weights_per_ch[ch] = []
            self.selections[ch] = PackedSelection()
            self.cutflows[ch] = {}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

        # trigger
        trigger = {}
        trigger_noiso = {}
        trigger_iso = {}
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

        # taus (will need to refine to avoid overlap with htt)
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        muons = ak.with_field(events.Muon, 0, "flavor")
        electrons = ak.with_field(events.Electron, 1, "flavor")

        # muons
        loose_muons = (
            (((muons.pt > 30) & (muons.pfRelIso04_all < 0.25)) | (muons.pt > 55))
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # electrons
        loose_electrons = (
            (((electrons.pt > 38) & (electrons.pfRelIso03_all < 0.25)) | (electrons.pt > 120))
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.cutBased >= electrons.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

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

        # get candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt

        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton
        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton
        mu_mvaId = candidatelep.mvaId if hasattr(candidatelep, "mvaId") else np.zeros(nevents)  # MVA-ID for candidate lepton
        mu_highPtId = ak.firsts(muons[good_muons]).highPtId
        ele_highPtId = ak.firsts(electrons[good_electrons]).cutBased_HEEP

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

        # for lep channel: first clean jets and leptons by removing overlap, then pick candidate_fj closest to the lepton
        lep_in_fj_overlap_bool = good_fatjets.delta_r(candidatelep_p4) > 0.1
        good_fatjets = good_fatjets[lep_in_fj_overlap_bool]
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # MET
        met = events.MET
        mt_lep_met = np.sqrt(
            2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )
        # delta phi MET and higgs candidate
        met_fjlep_dphi = candidatefj.delta_phi(met)

        # for leptonic channel: pick candidate_fj closest to the MET
        # candidatefj = ak.firsts(good_fatjets[ak.argmin(good_fatjets.delta_phi(met), axis=1, keepdims=True)])

        # fatjet - lepton mass
        lep_fj_mass = (candidatefj - candidatelep_p4).mass  # mass of fatjet without lepton

        # fatjet + neutrino
        candidateNeutrino = get_neutrino_z(candidatefj, met)
        rec_higgs_mass = (candidatefj + candidateNeutrino).mass  # mass of fatjet with lepton + neutrino

        # b-jets
        # pick highest b score in opposite direction from signal and make cut to avoid tt background events producing bjets)
        dphi_jet_lepfj = abs(goodjets.delta_phi(candidatefj))
        bjets_away_lepfj = goodjets[dphi_jet_lepfj > np.pi / 2]
        bjets = goodjets  # not necessarily opposite hemisphere

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

        variables = {
            "lep": {
                "fj_pt": candidatefj.pt,
                "fj_msoftdrop": candidatefj.msdcorr,
                "fj_bjets_ophem": ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1),
                "fj_bjets": ak.max(bjets.btagDeepFlavB, axis=1),
                "lep_pt": candidatelep.pt,
                "lep_isolation": lep_reliso,
                "lep_misolation": lep_miso,
                "lep_fj_m": lep_fj_mass,
                "lep_fj_dr": lep_fj_dr,
                "lep_met_mt": mt_lep_met,
                "met_fj_dphi": met_fjlep_dphi,
                "rec_higgs_m": rec_higgs_mass,
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
                "nj": n_jets_outside_ak8,
                "deta": deta,
                "mjj": mjj,
            },
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

            variables["common"]["hem_cleaning"] = hem_cleaning

        # gen-level matching
        if self.isMC:
            if ("HToWW" in dataset) or ("HWW" in dataset) or ("ttHToNonbb" in dataset):
                genVars, signal_mask = match_H(events.GenPart, candidatefj)
                self.add_selection(name="signal", sel=signal_mask)
            elif "HToTauTau" in dataset:
                genVars, signal_mask = match_H(events.GenPart, candidatefj, dau_pdgid=15)
                self.add_selection(name="signal", sel=signal_mask)
            elif ("WJets" in dataset) or ("ZJets" in dataset) or ("DYJets" in dataset):
                genVars = match_V(events.GenPart, candidatefj)
            elif "TT" in dataset:
                genVars = match_Top(events.GenPart, candidatefj)
            else:
                genVars = {}
            variables["common"] = {**variables["common"], **genVars}

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

            add_lepton_weight(self.weights, candidatelep, self._year + self._yearmod, "muon")
            add_lepton_weight(self.weights, candidatelep, self._year + self._yearmod, "electron")

            # self._btagSF.addBtagWeight(bjets_away_lepfj, self.weights, "lep")

            add_VJets_kFactors(self.weights, events.GenPart, dataset)

            # store the final common weight
            variables["common"]["weight"] = self.weights.partial_weight(self.common_weights)

            for key in self.weights._weights.keys():
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
                variables[varkey][f"weight_{key}"] = self.weights.partial_weight([key])
                if varkey in self.weights_per_ch.keys():
                    self.weights_per_ch[varkey].append(key)

            # store the per channel weight
            for ch in self._channels:
                if len(self.weights_per_ch[ch]) > 0:
                    variables[ch][f"weight_{ch}"] = self.weights.partial_weight(self.weights_per_ch[ch])

            # NOTE: to add variations:
            # for var in self.weights.variations:
            #     variables["common"][f"weight_{key}"] = self.weights.weight(key)

        """
        Selection and cutflows.
        """

        if self.apply_selection:
            if self.apply_trigger:
                for ch in self._channels:
                    self.add_selection(name="trigger", sel=trigger[ch], channel=[ch])
            self.add_selection(name="metfilters", sel=metfilters)
            self.add_selection(
                name="leptonKin", sel=(candidatelep.pt > 30), channel=["mu"]
            )
            self.add_selection(
                name="leptonKin", sel=(candidatelep.pt > 40), channel=["ele"]
            )
            self.add_selection(name="fatjetKin", sel=candidatefj.pt > 200)
            self.add_selection(name="ht", sel=(ht > 200))
            self.add_selection(
                name="oneLepton",
                sel=(n_good_muons == 1)
                & (n_good_electrons == 0)
                & (n_loose_electrons == 0)
                & ~ak.any(loose_muons & ~good_muons, 1),
                channel=["mu"],
            )
            self.add_selection(
                name="oneLepton",
                sel=(n_good_muons == 0)
                & (n_loose_muons == 0)
                & (n_good_electrons == 1)
                & ~ak.any(loose_electrons & ~good_electrons, 1),
                channel=["ele"],
            )
            self.add_selection(
                name="notaus", sel=(n_loose_taus_mu == 0), channel=["mu"]
            )
            self.add_selection(
                name="notaus", sel=(n_loose_taus_ele == 0), channel=["ele"]
            )
            self.add_selection(name="leptonInJet", sel=(lep_fj_dr < 0.8))
            # self.add_selection(
            #     name="antibjettag",
            #     # sel=(ak.max(bjets_away_lepfj.btagDeepFlavB, axis=1) < self._btagWPs["M"])
            #     sel=(ak.max(bjets.btagDeepFlavB, axis=1) < self._btagWPs["M"])
            # )

        # initialize pandas dataframe
        output = {}

        for ch in self._channels:
            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
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
                        # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                        # fill out dictionary
                        out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}

                # fill inference
                if self.inference:
                    for model_name in [
                        # "particlenet_hww_inclv2_pre2_noreg",
                        "ak8_MD_vminclv2ParT_manual_fixwrap",
                    ]:
                        pnet_vars = runInferenceTriton(
                            self.tagger_resources_path,
                            events[selection_ch],
                            fj_idx_lep[selection_ch],
                            model_name=model_name,
                        )

                        output[ch] = {
                            **output[ch],
                            **{key: value for (key, value) in pnet_vars.items()},
                        }

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

            if "rec_higgs_m" in output[ch].keys():
                output[ch]["rec_higgs_m"] = np.nan_to_num(output[ch]["rec_higgs_m"], nan=-1)

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
