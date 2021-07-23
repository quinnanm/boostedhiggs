import numpy as np
import awkward as ak
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

def match_HWWlepqq(genparticles,candidatefj):
    """
    return the number of matched objects (hWW*),daughters, 
    and gen flavor (enuqq, munuqq, taunuqq) 
    """
    # all the higgs bosons in the event (pdgID=25)
    higgs = getParticles(genparticles,25)
    
    # select our higgs to be all WW decays
    is_hWW = ak.all(abs(higgs.children.pdgId)==24,axis=2)
    higgs = higgs[is_hWW]
    
    # select higgs's children
    higgs_wstar = higgs.children[ak.argmin(higgs.children.mass,axis=2,keepdims=True)]
    higgs_w = higgs.children[ak.argmax(higgs.children.mass,axis=2,keepdims=True)]
    
    # select electrons, muons and taus
    prompt_electron = getParticles(genparticles,11,11,['isPrompt','isLastCopy'])
    prompt_muon = getParticles(genparticles,13,13,['isPrompt', 'isLastCopy'])
    prompt_tau = getParticles(genparticles,15,15,['isPrompt', 'isLastCopy'])
    
    # choosing the quarks that not only are coming from a hard process (e.g. the WW decay)
    # but also the ones whose parent is a W (pdgId=24), this avoids select quarks whose parent is a gluon 
    # who also happened to be produced in association with the Higgs
    prompt_q = getParticles(genparticles,0,5,['fromHardProcess', 'isLastCopy'])
    prompt_q = prompt_q[abs(prompt_q.distinctParent.pdgId) == 24]
    
    # counting the number of gen particles 
    n_electrons = ak.sum(prompt_electron.pt>0,axis=1)
    n_muons = ak.sum(prompt_muon.pt>0,axis=1)
    n_taus = ak.sum(prompt_tau.pt>0,axis=1)
    n_quarks = ak.sum(prompt_q.pt>0,axis=1)
    
    # we define the `flavor` of the Higgs decay
    # 4(elenuqq),6(munuqq),8(taunuqq)
    hWWlepqq_flavor = (n_quarks==2)*1 + (n_electrons==1)*3 + (n_muons==1)*5 + (n_taus==1)*7

    # since our jet has a cone size of 0.8, we use 0.8 as a dR threshold
    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)
    matchedW = candidatefj.nearest(higgs_w, axis=1, threshold=0.8)
    matchedWstar = candidatefj.nearest(higgs_wstar, axis=1, threshold=0.8) 

    # matched objects
    # 1 (H only), 4(W), 6(W star), 9(H, W and Wstar)
    hWWlepqq_matched = (ak.sum(matchedH.pt>0,axis=1)==1)*1 + (ak.sum(ak.flatten(matchedW.pt>0,axis=2),axis=1)==1)*3 + (ak.sum(ak.flatten(matchedWstar.pt>0,axis=2),axis=1)==1)*5    
    
    # let's concatenate all the daughters
    dr_fj_quarks = candidatefj.delta_r(prompt_q)
    dr_fj_electrons = candidatefj.delta_r(prompt_electron)
    dr_fj_muons = candidatefj.delta_r(prompt_muon)
    dr_fj_taus = candidatefj.delta_r(prompt_tau)
    dr_daughters = ak.concatenate([dr_fj_quarks,dr_fj_electrons,dr_fj_muons,dr_fj_taus],axis=1)
    
    #  number of visible daughters
    hWWlepqq_nprongs = ak.sum(dr_daughters<0.8,axis=1)
    
    return hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs,matchedH

class HwwSignalProcessor(processor.ProcessorABC):
    def __init__(self,jet_arbitration='met'):
        self._jet_arbitration = jet_arbitration
        self._regions = [
            "hadlep_signal",
            "hadmu_signal",
            "hadel_signal",
            "hadel_first",
            "hadel_second",
            "hadel_third",
        ]
        
        # output
        self.make_output = lambda: {
            'sumw': 0.,
            'signal_kin': hist2.Hist(
                hist2.axis.IntCategory([0, 2, 4, 6, 8], name='genflavor', label='gen flavor'),
                hist2.axis.IntCategory([0, 1, 4, 6, 9], name='genHflavor', label='higgs matching'),
                hist2.axis.Regular(100, 200, 1200, name='pt', label=r'Jet $p_T$ [GeV]'),
                hist2.axis.IntCategory([0, 1, 2, 3, 4], name='nprongs', label='Jet nprongs'),
                hist2.storage.Weight(),
            ),
            "lep_kin": hist2.Hist(
                hist2.axis.StrCategory(self._regions, name="region", label="Region"),
                hist2.axis.Regular(25, 0, 1, name="lepminiIso", label="lep miniIso"),
                hist2.axis.Regular(25, 0, 1, name="leprelIso", label="lep Rel Iso"),
                hist2.axis.Regular(40, 10, 800, name='lep_pt', label=r'lep $p_T$ [GeV]'),
                hist2.axis.Regular(50, 10, 1000, name='higgs_pt', label=r'matchedH $p_T$ [GeV]'),
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
    
        # match HWWlepqq 
        hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs,matchedH = match_HWWlepqq(events.GenPart,candidatefj)
        
        
        # select only leptons inside the jet
        dr_lep_jet_cut = candidatefj.delta_r(candidatelep_p4) < 0.8
        dr_lep_jet_cut = ak.fill_none(dr_lep_jet_cut, False)
        selection.add("dr_lep_jet", dr_lep_jet_cut)
        
        # select events with only electrons or muons
        selection.add('onemuon', (nmuons == 1) & (nlowptmuons <= 1) & (nelectrons == 0) & (nlowptelectrons == 0) & (ntaus == 0))
        selection.add('oneelectron', (nelectrons == 1) & (nlowptelectrons <= 1) & (nmuons == 0) & (nlowptmuons == 0) & (ntaus == 0))
            
        # select matched higgs bins
        higgspt = ak.firsts(matchedH.pt)
        selection.add("hfirst", (higgspt > 200) & (higgspt < 300))
        selection.add("hsecond", (higgspt > 300) & (higgspt < 350))
        selection.add("hthird", (higgspt > 350))

        regions = {
            "hadlep_signal": ["dr_lep_jet"],
            "hadmu_signal": ["onemuon", "dr_lep_jet"],
            "hadel_signal": ["oneelectron", "dr_lep_jet"],
            "hadel_first": ["oneelectron", "hfirst", "dr_lep_jet"],
            "hadel_second": ["oneelectron", "hsecond", "dr_lep_jet"],
            "hadel_third": ["oneelectron", "hthird", "dr_lep_jet"],
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
                lepminiIso=normalize(lep_miniIso, cut),
                leprelIso=normalize(lep_relIso, cut),
                lep_pt = normalize(candidatelep.pt, cut),
                higgs_pt=normalize(higgspt, cut),
                weight=weights.weight()[cut],
            )
            
        for region in regions:
            fill(region)
            
        # signal kin
        output['signal_kin'].fill(
            genflavor=normalize(hWWlepqq_flavor,dr_lep_jet_cut),
            genHflavor=normalize(hWWlepqq_matched,dr_lep_jet_cut),
            pt = normalize(candidatefj.pt,dr_lep_jet_cut),
            nprongs = normalize(hWWlepqq_nprongs,dr_lep_jet_cut),
            weight=weights.weight()[dr_lep_jet_cut],
        )
        
        return {dataset: output}
            
    def postprocess(self, accumulator):
        return accumulator
