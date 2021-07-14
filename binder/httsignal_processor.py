import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
hep.style.use(hep.style.CMS)

import hist as hist2
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights


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
                hist2.axis.Regular(100, 200, 1200, name='pt', label=r'Jet $p_T$'),
                hist2.storage.Weight(),
            )
        }
        
    def process(self, events):
        dataset = events.metadata['dataset']
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
        
        # function to normalize arrays after a cut or selection
        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        # select only leptons within the jet
        dr_lep_jet_cut = candidatefj.delta_r(candidatelep_p4) < 0.8
        dr_lep_jet_cut = ak.fill_none(dr_lep_jet_cut, False)
        
        # here we fill our histograms
        output['signal_kin'].fill(
            genflavor=normalize(htt_flavor,dr_lep_jet_cut),
            genHflavor=normalize(htt_matched,dr_lep_jet_cut),
            pt = normalize(candidatefj.pt,dr_lep_jet_cut),
            weight=weights.weight()[dr_lep_jet_cut],
        )

        return {dataset: output}
            

    def postprocess(self, accumulator):
        return accumulator
        

# Dask client        
from dask.distributed import Client

client = Client("tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.coffea.casa:8786")
client


# executing the processor for all arbitrations

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")

fileset = {
    "Htt": ["root://xcache/" + file for file in np.loadtxt("httdata.txt", dtype=str)] 
}

for arbitration in ["pt", "met", "lep"]:
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=HttSignalProcessor(jet_arbitration=arbitration),
        executor=processor.dask_executor,
        executor_args={
            "schema": processor.NanoAODSchema,
            "client": client,
        },
        maxchunks=30,
    )
    
    title = f"arbitration: {'$p_T$' if arbitration=='pt' else arbitration}"
    
    # hWWlepqq_matched
    gen_Hflavor = out["Htt"]["signal_kin"][{"genflavor": sum, "pt": sum}]
    match = ["None", r"$H$", r"$Hl$", r"$Hll$"]

    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )
    gen_Hflavor.plot1d(
        ax=ax,
        histtype="fill",
        density=True
    )
    ax.set(
        title=title,
        ylabel="Events",
        xlabel="Matched jets",
        xticklabels=match
    )
    fig.savefig(f"htt_matched_{arbitration}.png")
    
    # hWWlepqq_flavor
    gen_flavor = out["Htt"]["signal_kin"][{"genHflavor": sum, "pt": sum}]

    fig, ax = plt.subplots(
        figsize=(8,7),
        constrained_layout=True
    )
    gen_flavor.plot1d(
        ax=ax,
        density=True,
        histtype="fill",
    )
    ax.set(
        title=title,
        ylabel="Events",
        xlabel="gen flavor",
        xticklabels=["None",r"$\tau \tau$", r"$\tau e$", r"$\tau \mu$",r"$ee$",r"$\mu \mu$"]
    )
    fig.savefig(f"htt_genflavor_{arbitration}.png")
    
    # hWWlepqq_matched and jet pt
    h = out["Htt"]["signal_kin"][{"genflavor":sum}]

    fig, ax = plt.subplots(
        figsize=(10,7),
        constrained_layout=True
    )
    for i in range(4): 
        h[i,:].plot1d(ax=ax)
        
    ax.set(
        title=title,
        ylabel="Events",
        xlim=(180,600),
        xlabel="jet $p_T$ [GeV]"
    )
    ax.legend(match, title="matched")
    fig.savefig(f"htt_matchvspt_{arbitration}.png")
