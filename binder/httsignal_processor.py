import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
hep.style.use(hep.style.CMS)

import hist as hist2
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection


def getParticles(genparticles,lowid=22,highid=25,flags=['fromHardProcess', 'isLastCopy']):
    """
    returns the particle objects that satisfy a low id, 
    high id condition and have certain flags
    """
    absid = abs(genparticles.pdgId)
    return genparticles[
        ((absid >= lowid) & (absid <= highid))
        & genparticles.hasFlags(flags)
    ]

def match_Htt(events,candidatefj):   
    """
    return the number of matched objects (H, Hlep, Hleplep), 
    and gen flavor (tt, te, tmu, ee, mumu) 
    """
    higgs = getParticles(events.GenPart,25)
    
    # select all Higgs bosons that decay into taus
    is_htt = ak.all(abs(higgs.children.pdgId)==15,axis=2)
    higgs = higgs[is_htt]
    
    # electrons and muons coming from taus
    fromtau_electron = getParticles(events.GenPart,11,11,['isDirectTauDecayProduct'])
    fromtau_muon = getParticles(events.GenPart,13,13,['isDirectTauDecayProduct'])
    
    n_electrons_fromtaus = ak.sum(fromtau_electron.pt>0,axis=1)
    n_muons_fromtaus = ak.sum(fromtau_muon.pt>0,axis=1)
    
    # visible gen taus
    tau_visible = events.GenVisTau
    n_visibletaus = ak.sum(tau_visible.pt>0,axis=1)
    
    # 3(tt), 6(te), 8(tmu), 10(ee), 12(mumu)
    htt_flavor = (n_visibletaus==1)*1 + (n_visibletaus==2)*3 + (n_electrons_fromtaus==1)*5 + (n_muons_fromtaus==1)*7 + (n_electrons_fromtaus==2)*10 + (n_muons_fromtaus==2)*12

    # we need to guarantee that both of the taus are inside of the jet cone
    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)
    dr_fj_visibletaus = candidatefj.delta_r(tau_visible)
    dr_fj_electrons = candidatefj.delta_r(fromtau_electron)
    dr_fj_muons = candidatefj.delta_r(fromtau_muon)
    dr_daughters = ak.concatenate([dr_fj_visibletaus,dr_fj_electrons,dr_fj_muons],axis=1)
    
    # 1 (H only), 4 (H and one tau/electron or muon from tau), 6 (H and 2 taus/ele/mu)
    htt_matched = (ak.sum(matchedH.pt>0,axis=1)==1) + (ak.sum(dr_daughters<0.8,axis=1)==1)*3 + (ak.sum(dr_daughters<0.8,axis=1)==2)*5 
    
    return htt_flavor,htt_matched


class HttSignalProcessor(processor.ProcessorABC):
    def __init__(self,jet_arbitration='pt'):
        self._jet_arbitration = jet_arbitration
        
        # output
        self.make_output = lambda: {
            'sumw': 0.,
            'signal_kin': hist2.Hist(
                hist2.axis.IntCategory([0, 3, 6, 8, 10, 12], name='genflavor', label='gen flavor'),
                hist2.axis.IntCategory([0, 1, 4, 6], name='genHflavor', label='higgs matching'),
                hist2.axis.Regular(100, 200, 1200, name='pt', label=r'Jet $p_T$ [GeV]'),
                hist2.storage.Weight(),
            ),
            "lep_kin": hist2.Hist(
                hist2.axis.StrCategory(["hadmu_signal", "hadel_signal"], name="region", label="Region"),
                hist2.axis.Regular(25, 0, 1, name="lepminiIso", label="lep miniIso"),
                hist2.axis.Regular(25, 0, 1, name="leprelIso", label="lep Rel Iso"),
                hist2.axis.Regular(40, 10, 800, name='lep_pt', label=r'lep $p_T$ [GeV]'),
                hist2.storage.Weight(),
            ),
        }
        
    def process(self, events):
        dataset = events.metadata['dataset']
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        weights.add('genweight', events.genWeight)
        
        output = self.make_output()
        output['sumw'] = ak.sum(events.genWeight)
            
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
        
        goodtau = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEle >= 8)
            & (events.Tau.idAntiMu >= 1)
        )
        ntaus = ak.sum(goodtau, axis=1)

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
        
        # lepton isolation
        lep_miniIso = candidatelep.miniPFRelIso_all
        lep_relIso = candidatelep.pfRelIso03_all
        
        # met
        met = events.MET
        
        # jets
        fatjets = events.FatJet
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
            
        # match and flavor htt 
        htt_flavor,htt_matched = match_Htt(events,candidatefj)
        
        # select events with only electrons or muons
        selection.add('onemuon', (nmuons == 1) & (nlowptmuons <= 1) & (nelectrons == 0) & (nlowptelectrons == 0) & (ntaus == 0))
        selection.add('oneelectron', (nelectrons == 1) & (nlowptelectrons <= 1) & (nmuons == 0) & (nlowptmuons == 0) & (ntaus == 0))
            
        # select only leptons inside the jet
        dr_lep_jet_cut = candidatefj.delta_r(candidatelep_p4) < 0.8
        dr_lep_jet_cut = ak.fill_none(dr_lep_jet_cut, False)
        selection.add("dr_lep_jet", dr_lep_jet_cut)
        
        regions = {
            "hadmu_signal": ["onemuon", "dr_lep_jet"],
            "hadel_signal": ["oneelectron", "dr_lep_jet"]
        }
        
        # function to normalize arrays after a cut or selection
        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        
	# lepton kin
        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)

            output['lep_kin'].fill(
                region=region,
                lepminiIso=normalize(lep_miniIso,cut),
                leprelIso=normalize(lep_relIso,cut),
                lep_pt = normalize(candidatelep.pt,cut),
                weight=weights.weight()[cut],
            )
            
        for region in regions:
            fill(region)
        
        # signal kin
        output['signal_kin'].fill(
            genflavor=normalize(htt_flavor, dr_lep_jet_cut),
            genHflavor=normalize(htt_matched,dr_lep_jet_cut),
            pt = normalize(candidatefj.pt,dr_lep_jet_cut),
            weight=weights.weight()[dr_lep_jet_cut],
        )

        return {dataset: output}
            

    def postprocess(self, accumulator):
        return accumulator
