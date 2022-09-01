import json

"""
Cross Sections

Jennet has a nice notebook on x-sections:
https://github.com/jennetd/hbb-coffea/blob/master/xsec-json.ipynb
"""

# Branching ratio of H to WW (LHC-XS-WG)
BR_HWW = 0.2137
# Branching ratio of WW to LNuQQ (0.21664272)
BR_WW_lnuqq = (0.1046+0.1050+0.1075)*(0.6832)
# Branching ratio of WW to 4Q (0.46676224)
BR_WW_qqqq = 0.6832**2

BR_THadronic = 0.665
BR_TLeptonic = 1 - BR_THadronic

BR_HBB = 0.5809
BR_Htt = 0.06272

xs = {}

# GluGluToHToWWPt200
# - From powheg: 0.4716 pb
# - PKU uses 0.1014pb after BR
xs["GluGluHToWW_Pt-200ToInf_M-125"] = 0.4716*BR_HWW

# GluGluHToWWToLNuQQ
# - From GenXsecAnalyzer: 28.87 pb
#    - Times BR: 28.87*0.2137*0.21664272 = 1.336 pb
#  - From LHC-XS-WG: 
#    - ggF: 48.58 pb (N3LO)
#           44.14 pb (NNLO + NNLL)
#      - Times BR: 48.58*0.2137*0.21664272 = 2.2491 pb
xs["GluGluHToWWToLNuQQ"] = 48.58 * BR_HWW * BR_WW_lnuqq

# VBFHToWWToLNuQQ-MH125
#  - From GenXsecAnalyzer: 3.892 pb
#  - From LHC-XS-WG: 
#    - VBF: 3.782E pb (NNLO QCD + NLO EW)
#      - Times BR: 3.782*0.2137*0.21664272 = 0.1751 pb
xs["VBFHToWWToLNuQQ-MH125"] = 3.782 * BR_HWW * BR_WW_lnuqq

## Note: will need to add DipoleRecoilOn sample at some point
# xs["VBFHToWWDipoleRecoilOn"] = xs["VBFHToWWToLNuQQ-MH125"]

xs["ttHToNonbb_M125"] = 5.013e-01 * (1-BR_HBB)

# Cross xcheck the following numbers
xs["HWminusJ_HToWW_M-125"] = 0.5445 * BR_HWW
xs["HWplusJ_HToWW_M-125"] = 0.8720 * BR_HWW
xs["HZJ_HToWW_M-125"] = 0.9595 * BR_HWW

# QCD
xs["QCD_HT500to700"] = 3.033e+04
xs["QCD_HT700to1000"] = 6.412e+03
xs["QCD_HT1000to1500"] = 1.118e+03
xs["QCD_HT1500to2000"] = 1.085e+02
xs["QCD_HT2000toInf"] = 2.194e+01

xs["QCD_Pt_120to170"] = 4.074e+05
xs["QCD_Pt_170to300"] = 1.035e+05
xs["QCD_Pt_300to470"] = 6.833e+03
xs["QCD_Pt_470to600"] = 5.495e+02
xs["QCD_Pt_600to800"] = 156.5
xs["QCD_Pt_800to1000"] = 2.622e+01
xs["QCD_Pt_1000to1400"] = 7.475e+00
xs["QCD_Pt_1400to1800"] = 6.482e-01
xs["QCD_Pt_1800to2400"] = 8.742e-02
xs["QCD_Pt_2400to3200"] = 5.237e-03
xs["QCD_Pt_3200toInf"] = 1.353e-04

# TTbar
xs["TTTo2L2Nu"] = 6.871e+02 * BR_TLeptonic**2
xs["TTToHadronic"] = 6.871e+02 * BR_THadronic**2
xs["TTToSemiLeptonic"] = 6.871e+02 * 2 * BR_TLeptonic * BR_THadronic

# Single Top
xs["ST_s-channel_4f_leptonDecays"] = 3.549e+00 * BR_TLeptonic
xs["ST_t-channel_antitop_4f_InclusiveDecays"] = 6.793e+01
xs["ST_t-channel_antitop_5f_InclusiveDecays"] = 7.174e+01
xs["ST_t-channel_top_4f_InclusiveDecays"] = 1.134e+02
xs["ST_t-channel_top_5f_InclusiveDecays"] = 1.197e+02
xs["ST_tW_antitop_5f_inclusiveDecays"] = 3.251e+01
xs["ST_tW_antitop_5f_NoFullyHadronicDecays"] = 3.251e+01 * BR_TLeptonic
xs["ST_tW_top_5f_inclusiveDecays"] = 3.245e+01
xs["ST_tW_top_5f_NoFullyHadronicDecays"] = 3.245e+01 * BR_TLeptonic

# W+jets W(qq)
xs["WJetsToQQ_HT-400to600"] = 2.770e+02 
xs["WJetsToQQ_HT-600to800"] = 5.906e+01 
xs["WJetsToQQ_HT-800toInf"] = 2.875e+01 
    
# W+jets W(lv)
xs["WJetsToLNu_HT-70To100"] = 1.270e+03
xs["WJetsToLNu_HT-100To200"] = 1.252e+03 
xs["WJetsToLNu_HT-200To400"] = 3.365e+02 
xs["WJetsToLNu_HT-400To600"] = 4.512e+01 
xs["WJetsToLNu_HT-600To800"] = 1.099e+01 
xs["WJetsToLNu_HT-800To1200"] = 4.938e+00 
xs["WJetsToLNu_HT-1200To2500"] = 1.155e+00 
xs["WJetsToLNu_HT-2500ToInf"] = 2.625e-02 

# Z+jets Z(qq)
xs["ZJetsToQQ_HT-400to600"] = 1.145e+02
xs["ZJetsToQQ_HT-600to800"] = 2.541e+01
xs["ZJetsToQQ_HT-800toInf"] = 1.291e+01
        
# DY+jets
xs["DYJetsToLL_HT-70to100"] = 1.399e+02
xs["DYJetsToLL_HT-100to200"] = 1.401e+02
xs["DYJetsToLL_HT-200to400"] = 3.835e+01
xs["DYJetsToLL_HT-400to600"] = 5.217e+00
xs["DYJetsToLL_HT-600to800"] = 1.267e+00
xs["DYJetsToLL_HT-800to1200"] = 5.682e-01
xs["DYJetsToLL_HT-1200to2500"] = 1.332e-01
xs["DYJetsToLL_HT-2500toInf"] = 2.978e-03

xs["DYJetsToLL_Pt-50To100"] = 3.941e+02
xs["DYJetsToLL_Pt-100To250"] = 9.442e+01
xs["DYJetsToLL_Pt-250To400"] = 3.651e+00
xs["DYJetsToLL_Pt-400To650"] = 4.986e-01
xs["DYJetsToLL_Pt-650ToInf"] = 4.678e-02

# VV 
xs["WW"] = 7.583e+01
xs["WZ"] = 2.756e+01
xs["ZZ"] = 1.214e+01
xs["WWTo1L1Nu2Q"] = 5.090e+01
xs["WWTo4Q"] = 5.157e+01
xs["WZTo1L1Nu2Q"] = 9.152e+00
xs["WZTo2Q2L"] = 6.422e+00
xs["ZZTo2Q2L"] = 3.705e+00
xs["ZZTo2Q2Nu"] = 4.498e+00
xs["ZZTo4Q"] = 3.295e+00

# EWK Z
xs["EWKZ_ZToQQ"] = 9.791e+00
xs["EWKZ_ZToLL"] = 6.207e+00
xs["EWKZ_ZToNuNu"] = 1.065e+01

# EWK W
xs["EWKWminus_WToQQ"] = 1.917e+01
xs["EWKWplus_WToQQ"] = 2.874e+01
xs["EWKWminus_WToLNu"] = 3.208e+01
xs["EWKWplus_WToLNu"] = 3.909e+01

# Higgs to BB
xs["GluGluHToBB"] = 4.716e-01 * BR_HBB
xs["VBFHToBB"] = 3.873e+00 * BR_HBB
xs["VBFHToBBDipoleRecoilOn"] = xs["VBFHToBB"]

xs["WminusH_HToBB_WToQQ"] = 3.675e-01 * BR_HBB
xs["WplusH_HToBB_WToQQ"] = 5.890e-01 * BR_HBB
xs["WminusH_HToBB_WToLNu"] = 1.770e-01 * BR_HBB
xs["WplusH_HToBB_WToLNu"] = 2.832e-01 * BR_HBB

xs["ZH_HToBB_ZToQQ"] = 5.612e-01 * BR_HBB
xs["ZH_HToBB_ZToLL"] = 7.977e-02 * BR_HBB
xs["ZH_HToBB_ZToNuNu"] = 1.573e-01 * BR_HBB

xs["ggZH_HToBB_ZToNuNu"] = 1.222e-02 * BR_HBB
xs["ggZH_HToBB_ZToQQ"] = 4.319e-02 * BR_HBB
xs["ggZH_HToBB_ZToLL"] = 6.185e-03 * BR_HBB

xs["ttHToBB"] = 5.013e-01 * BR_HBB

xs["GluGluHToTauTau"]  = 48.58 * BR_Htt

print(xs)

with open("xsec_pfnano.json", "w") as outfile:
    json.dump(xs, outfile, indent=4)
