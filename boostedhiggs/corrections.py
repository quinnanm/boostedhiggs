import os
import numpy as np
import awkward as ak
import gzip
import pickle
from coffea.lookup_tools.lookup_base import lookup_base

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
print(DATA_DIR)
with gzip.open(os.path.join(DATA_DIR, 'corrections.pkl.gz')) as fin:
    compiled = pickle.load(fin)

compiled['2017_pileupweight']._values = np.minimum(5, compiled['2017_pileupweight']._values)
compiled['2018_pileupweight']._values = np.minimum(5, compiled['2018_pileupweight']._values)

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

def corrected_msoftdrop(fatjets, subjets):
    sf = _softdrop_weight(fatjets.pt, fatjets.eta)
    sf = np.maximum(1e-5, sf)
    dazsle_msd = (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum()
    return dazsle_msd.mass * sf

def add_pileup_weight(weights, nPU, year='2017', dataset=None):
    if year == '2017' and dataset in compiled['2017_pileupweight_dataset']:
        weights.add(
            'pileup_weight',
            compiled['2017_pileupweight_dataset'][dataset](nPU),
            compiled['2017_pileupweight_dataset_puUp'][dataset](nPU),
            compiled['2017_pileupweight_dataset_puDown'][dataset](nPU),
        )
    else:
        weights.add(
            'pileup_weight',
            compiled[f'{year}_pileupweight'](nPU),
            compiled[f'{year}_pileupweight_puUp'](nPU),
            compiled[f'{year}_pileupweight_puDown'](nPU),
        )
