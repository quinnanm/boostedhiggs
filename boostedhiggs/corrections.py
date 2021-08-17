import os
import numpy as np
import awkward as ak
import gzip
import pickle
import correctionlib
from coffea.lookup_tools.lookup_base import lookup_base
from coffea import lookup_tools
from coffea import util

with gzip.open(os.path.join(os.path.dirname(__file__), 'data', 'corrections.pkl.gz')) as fin:
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

def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask
    path = os.path.join(os.path.dirname(__file__), 'data', filename)
    return LumiMask(path)

lumiMasks = {
    '2016': build_lumimask('Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'),
    '2017': build_lumimask('Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt'),
    '2018': build_lumimask('Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'),
}

def add_pileup_weight(weights, nPU, year='2017', dataset=None):
    weights.add(
        'pileup_weight',
        compiled[f'{year}_pileupweight'](nPU),
        compiled[f'{year}_pileupweight_puUp'](nPU),
        compiled[f'{year}_pileupweight_puDown'](nPU),
    )
