import os
import numpy as np
import awkward as ak
import gzip
import pickle
import importlib.resources
import correctionlib
from coffea.lookup_tools.lookup_base import lookup_base
from coffea import lookup_tools
from coffea import util

with importlib.resources.path("boostedhiggs.data", "corrections.pkl.gz") as path:
    with gzip.open(path) as fin:
        compiled = pickle.load(fin)

class SoftDropWeight(lookup_base):
    def _evaluate(self, pt, eta):
        gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
        cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
        fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
        genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
        ptpow = np.power.outer(pt, np.arange(cpar.size))
        cenweight = np.dot(ptpow, cpar)
        forweight = np.dot(ptpow, fpar)
        weight = np.where(np.abs(eta) < 1.3, cenweight, forweight)
        return genw*weight

_softdrop_weight = SoftDropWeight()

def corrected_msoftdrop(fatjets):
    sf = _softdrop_weight(fatjets.pt, fatjets.eta)
    sf = np.maximum(1e-5, sf)
    dazsle_msd = (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum()
    return dazsle_msd.mass * sf

def add_pileup_weight(weights, nPU, year='2017'):
    weights.add(
        'pileup_weight',
        compiled[f'{year}_pileupweight'](nPU),
        compiled[f'{year}_pileupweight_puUp'](nPU),
        compiled[f'{year}_pileupweight_puDown'](nPU),
    )

def add_pdf_weight(weights, pdf_weights):
    nom = np.ones(len(weights.weight()))
    up = np.ones(len(weights.weight()))
    down = np.ones(len(weights.weight()))

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
    if pdf_weights is not None and "306000 - 306102" in pdf_weights.__doc__:
        # Hessian PDF weights
        # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        arg = pdf_weights[:, 1:-2] - np.ones((len(weights.weight()), 100))
        summed = ak.sum(np.square(arg), axis=1)
        pdf_unc = np.sqrt((1. / 99.) * summed)
        weights.add('PDF_weight', nom, pdf_unc + nom)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5 * (pdf_weights[:, 102] - pdf_weights[:, 101])
        weights.add('aS_weight', nom, as_unc + nom)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        weights.add('PDFaS_weight', nom, pdfas_unc + nom)

    else:
        weights.add('aS_weight', nom, up, down)
        weights.add('PDF_weight', nom, up, down)
        weights.add('PDFaS_weight', nom, up, down)

def add_ps_weight(weights, ps_weights):
    nom = np.ones(len(weights.weight()))
    up_isr = np.ones(len(weights.weight()))
    down_isr = np.ones(len(weights.weight()))
    up_fsr = np.ones(len(weights.weight()))
    down_fsr = np.ones(len(weights.weight()))

    if ps_weights is not None:
        if len(ps_weights[0]) == 4:
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
        else:
            warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")
    weights.add('UEPS_ISR', nom, up_isr, down_isr)
    weights.add('UEPS_FSR', nom, up_fsr, down_fsr)

def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask
    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)

lumiMasks = {
    "2016": build_lumimask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
    "2017": build_lumimask("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
    "2018": build_lumimask("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
}

# TODO: Update to UL
# Option: 0 for eta-pt, 1 for abseta-pt, 2 for pt-abseta
lepton_sf_dict = {
    "elec_RECO": 0,
    "elec_ID": 0, 
    "elec_TRIG32": 0,
    "elec_TRIG115": 1,
    "muon_ISO": 1,
    "muon_ID": 1,
    "muon_TRIG27": 2,
    "muon_TRIG50": 2,
}

def add_leptonSFs(weights, lepton, year, match):
    for sf in lepton_sf_dict:
        sfoption = lepton_sf_dict[sf]
        lep_pt = np.array(ak.fill_none(lepton.pt, 0.))
        lep_eta = np.array(ak.fill_none(lepton.eta, 0.))
        lep_abseta = np.array(ak.fill_none(abs(lepton.eta), 0.))
        if match in sf:
            if sfoption==0:
                nom = compiled['%s_value'%sf](lep_eta,lep_pt)
                err = compiled['%s_error'%sf](lep_eta,lep_pt)
            elif sfoption==1:
                nom = compiled['%s_value'%sf](np.abs(lep_eta),lep_pt)
                err = compiled['%s_error'%sf](np.abs(lep_eta),lep_pt)
            elif sfoption==2:
                nom = compiled['%s_value'%sf](lep_pt,np.abs(lep_eta))
                err = compiled['%s_error'%sf](lep_pt,np.abs(lep_eta))
            else: 
                print('Error: Invalid type ordering for lepton SF %s'%sf)
                return
            if "TRIG27" in sf:
                nom[lep_pt>55.] = 1.
                err[lep_pt>55.] = 0.
            if "TRIG50" in sf:
                nom[lep_pt<55.] = 1.
                err[lep_pt<55.] = 0.
            if "TRIG32" in sf:
                nom[lep_pt>120.] = 1.
                err[lep_pt>120.] = 0.
            if "TRIG115" in sf:
                nom[lep_pt<120.] = 1.
                err[lep_pt<120.] = 0.
            weights.add(sf, nom, nom+err, nom-err)

def is_overlap(events,dataset,triggers,year):
    dataset_ordering = {
        '2016':['SingleMuon','SingleElectron','MET','JetHT'],
        '2017':['SingleMuon','SingleElectron','MET','JetHT'],
        '2018':['SingleMuon','EGamma','MET','JetHT']
    }
    pd_to_trig = {
        'SingleMuon': ['Mu50',
                       'Mu55',
                       'Mu15_IsoVVVL_PFHT600',
                       'Mu15_IsoVVVL_PFHT450_PFMET50',
                       ],
        'SingleElectron': ['Ele50_CaloIdVT_GsfTrkIdT_PFJet165',
                           'Ele115_CaloIdVT_GsfTrkIdT',
                           'Ele15_IsoVVVL_PFHT600',
                           'Ele35_WPTight_Gsf',
                           'Ele15_IsoVVVL_PFHT450_PFMET50',
                       ],
        'JetHT': ['PFHT800',
                  'PFHT900',
                  'AK8PFJet360_TrimMass30',
                  'AK8PFHT700_TrimR0p1PT0p03Mass50',
                  'PFHT650_WideJetMJJ950DEtaJJ1p5',
                  'PFHT650_WideJetMJJ900DEtaJJ1p5',
                  'PFJet450',
                  'PFHT1050',
                  'PFJet500',
                  'AK8PFJet400_TrimMass30',
                  'AK8PFJet420_TrimMass30',
                  'AK8PFHT800_TrimMass50'
              ],
        'MET': ['PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu110_PFMHTNoMu110_IDTight',
            ],
    }
    
    overlap = np.ones(len(events), dtype='bool')
    for p in dataset_ordering[year]:
        if dataset.startswith(p):
            pass_pd = np.zeros(len(events), dtype='bool')
            for t in pd_to_trig[p]:
                if t in events.HLT.fields:
                    pass_pd = pass_pd | events.HLT[t]
            overlap = overlap & pass_pd
            break
        else:
            for t in pd_to_trig[p]:
                if t in events.HLT.fields:
                    overlap = overlap & np.logical_not(events.HLT[t])
    return overlap
