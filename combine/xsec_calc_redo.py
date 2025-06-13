import os
import subprocess
import re
import json
import ROOT as rt
import numpy as np
rt.gStyle.SetOptStat(0)
import pandas as pd
import os
import pickle


def calcxsec(sumweightspass, sumweightstotal, lum, xs):
    calc = (sumweightspass / sumweightstotal ) * lum * xs
    print(calc)
    return calc

# Replace 'your_file.parquet' with the path to your Parquet file
years = ['2016','2016APV','2017','2018']
lumi = [16809.96, 19492.72, 41476.02, 59816.23]
xsecs = {'vbf': 0.8082134,
         'ggf': 0.10078092000000001}
totxsec_vbf = 0.0
sgw_tot_vbf = [0,0,0,0]
sgw_pass_vbf = [0,0,0,0]

totxsec_ggf_200_300 = 0.0
totxsec_ggf_300_450 = 0.0
totxsec_ggf_450_inf = 0.0

sgw_tot_ggf = [0,0,0,0]

sgw_pass_ggf_200_300 = [0,0,0,0]
sgw_pass_ggf_300_450 = [0,0,0,0]
sgw_pass_ggf_450_inf = [0,0,0,0]

#uncertainties
#QCDscale
scnames = ['weight_scale0', 'weight_scale1', 'weight_scale3', 'weight_scale5', 'weight_scale7', 'weight_scale8', 'weight_scale4']
scl_tot_vbf = {name: [0, 0, 0, 0] for name in scnames}
scl_reco_vbf = {name: [0, 0, 0, 0] for name in scnames}

scl_tot_ggf = {name: [0, 0, 0, 0] for name in scnames}
scl_reco_ggf_200_300 = {name: [0, 0, 0, 0] for name in scnames}
scl_reco_ggf_300_450 = {name: [0, 0, 0, 0] for name in scnames}
scl_reco_ggf_450_inf = {name: [0, 0, 0, 0] for name in scnames}

#VBF #######################
for i,year in enumerate(years):

    # Initialize total event count
    total_events = 0
    # sgw_tot_vbf = 0
    # sgw_pass_vbf = 0
    
    directory = '/eos/uscms/store/user/fmokhtar/boostedhiggs/Dec20_hww_stxs_'+year+'/VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet/outfiles/'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('ele.parquet'):
            # print('ELECTRON-------------------------------')
            df = pd.read_parquet(filepath)
            num_events = len(df)
            total_events += num_events
            #from https://github.com/farakiko/boostedhiggs/blob/main/binder/STXS.ipynb
            events_finecat = df[((df['STXS_finecat']%100 == 21) | (df['STXS_finecat']%100 == 22) | (df['STXS_finecat']%100 == 23) | (df['STXS_finecat']%100 == 24)) ]
            passsum = events_finecat['weight_ele_genweight'].sum()
            totsum = df['weight_ele_genweight'].sum()
            sgw_pass_vbf[i] += float(passsum)
            sgw_tot_vbf[i] += float(totsum)
            
            #uncertainty sums
            for name in scnames:
                scsum = df[name].sum()
                events_reco = df[((df['mjj']>1000) & (df['deta']>3.5) & (df['NumOtherJets']>=2)) ]
                scsum_reco = events_reco[name].sum()

                scl_tot_vbf[name][i] += float(scsum)
                scl_reco_vbf[name][i] += float(scsum_reco)
                
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as pklfile:
                data = pickle.load(pklfile)
                # print(list(data['VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'][year].keys()))
                sumgenweight = data['VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'][year]['sumgenweight']                
                # print(f"sumgenweight: {sumgenweight}")


    #calc xsec
    xsec_vbf = 0.0
    nev_vbf = 0
    #xsec from here https://github.com/farakiko/boostedhiggs/blob/main/fileset/xsec_pfnano.json
    if total_events>0:
        xsec_vbf = (sgw_pass_vbf[i] / sgw_tot_vbf[i] ) * lumi[i] * xsecs['vbf']
    else:
        print("WARNING: NO EVENTS")
                
    # Print the total number of events
    print(year)
    print('----------------------------------------')
    # print(f"VBF Total number of events: {total_events}")
    print(f"VBF sum_gen_weights_total_VBF: {sgw_tot_vbf[i]}")
    print(f"VBF sum_gen_weights_events_pass_genVBF: {sgw_pass_vbf[i]}")
    print(f"lumi: {lumi[i]}")
    print(f"VBF xsec: {xsec_vbf}")
    print('----------------------------------------')

print("==================TOTALS===========================")
print(f"years: {years}")
print(f"sgw_pass_vbf: {sgw_pass_vbf}")
print(f"sgw_tot_vbf: {sgw_tot_vbf}")
totxsec_vbf = (sum(sgw_pass_vbf) / sum(sgw_tot_vbf) )* sum(lumi) * xsecs['vbf']
print(f"VBF TOTAL xsec: {totxsec_vbf}")

print("----------UNCERTAINTIES----------")
print(f"scl_tot_vbf:")
for name, array in scl_tot_vbf.items():
    print(f"{name}: {array}")
print(f"scl_reco_vbf:")
for name, array in scl_reco_vbf.items():
    print(f"{name}: {array}")

#GGF ########################
# r_ggH_pt200_300
# r_ggH_pt300_450
# r_ggH_pt450_inf 

# tot_ggf = 0
# totxsec_ggf = 0.0


for i,year in enumerate(years):
    
    directory = '/eos/uscms/store/user/fmokhtar/boostedhiggs/Dec20_hww_stxs_'+year+'/GluGluHToWW_Pt-200ToInf_M-125_Rivet/outfiles/'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('ele.parquet'):
           # print(filename)
            df = pd.read_parquet(filepath)
            # print(list(df.columns))
            events_200_300 = df[((df['STXS_finecat']%100 == 1) | (df['STXS_finecat']%100 == 5))]
            events_300_450 = df[((df['STXS_finecat']%100 == 2) | (df['STXS_finecat']%100 == 6))]
            events_450_inf = df[((df['STXS_finecat']%100 == 3) | (df['STXS_finecat']%100 == 4) | (df['STXS_finecat']%100 == 7) | (df['STXS_finecat']%100 == 8))]

            passsum_200_300 = events_200_300['weight_ele_genweight'].sum()
            passsum_300_450 = events_300_450['weight_ele_genweight'].sum()
            passsum_450_inf = events_450_inf['weight_ele_genweight'].sum()

            totsum = df['weight_ele_genweight'].sum()
            sgw_pass_ggf_200_300[i] += float(passsum_200_300)
            sgw_pass_ggf_300_450[i] += float(passsum_300_450)
            sgw_pass_ggf_450_inf[i] += float(passsum_450_inf)
            sgw_tot_ggf[i] += float(totsum)

            #uncertainty sums
            for name in scnames:
                scsum = df[name].sum()

                events_reco_200_300 = df[(((df['mjj']<1000) | (df['deta']<3.5) | (df['NumOtherJets']<2)) & ((df['rec_higgs_pt']>250) & (df['rec_higgs_pt']<350)))]
                events_reco_300_450 = df[(((df['mjj']<1000) | (df['deta']<3.5) | (df['NumOtherJets']<2)) & ((df['rec_higgs_pt']>350) & (df['rec_higgs_pt']<500)))]
                events_reco_450_inf = df[(((df['mjj']<1000) | (df['deta']<3.5) | (df['NumOtherJets']<2)) & ((df['rec_higgs_pt']>500) & (df['rec_higgs_pt']<2500)))]
                scsum_reco_200_300 = events_reco_200_300[name].sum()
                scsum_reco_300_450 = events_reco_300_450[name].sum()
                scsum_reco_450_inf = events_reco_450_inf[name].sum()

                
                scl_tot_ggf[name][i] += float(scsum)
                scl_reco_ggf_200_300[name][i] += float(scsum_reco_200_300)
                scl_reco_ggf_300_450[name][i] += float(scsum_reco_300_450)
                scl_reco_ggf_450_inf[name][i] += float(scsum_reco_450_inf)

            
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as pklfile:
                data = pickle.load(pklfile)
#                 # print(data)
                sumgenweight = data['GluGluHToWW_Pt-200ToInf_M-125_Rivet'][year]['sumgenweight']
#                 #print(sumgenweight)
#                 weight_sum += float(sumgenweight)

    xsec_200_300 = (sgw_pass_ggf_200_300[i] / sgw_tot_ggf[i] ) * lumi[i] * xsecs['ggf']
    xsec_300_450 = (sgw_pass_ggf_300_450[i] / sgw_tot_ggf[i] ) * lumi[i] * xsecs['ggf']
    xsec_450_inf = (sgw_pass_ggf_450_inf[i] / sgw_tot_ggf[i] ) * lumi[i] * xsecs['ggf']

    
#     # Print the total number of events
    print(year)
    print('----------------------------------------')
    print(f"GGF sum_gen_weights_total : {sgw_tot_ggf[i]}")
    print(f"GGF sum_gen_weights_events_pass 200_300: {sgw_pass_ggf_200_300[i]}")
    print(f"GGF sum_gen_weights_events_pass 300_450: {sgw_pass_ggf_300_450[i]}")
    print(f"GGF sum_gen_weights_events_pass 450_inf: {sgw_pass_ggf_450_inf[i]}")
    print(f"lumi: {lumi[i]}")
    print(f"GGF xsec_200_300: {xsec_200_300}")
    print(f"GGF xsec_300_450: {xsec_300_450}")
    print(f"GGF xsec_450_inf: {xsec_450_inf}")
    print('----------------------------------------')


totxsec_ggf_200_300 = (sum(sgw_pass_ggf_200_300) / sum(sgw_tot_ggf) )* sum(lumi) * xsecs['ggf']
totxsec_ggf_300_450 = (sum(sgw_pass_ggf_300_450) / sum(sgw_tot_ggf) )* sum(lumi) * xsecs['ggf']
totxsec_ggf_450_inf = (sum(sgw_pass_ggf_450_inf) / sum(sgw_tot_ggf) )* sum(lumi) * xsecs['ggf']

print("==================TOTALS===========================")
print(f"years: {years}")
print(f"GGF TOTAL xsec 200_300: {totxsec_ggf_200_300}")
print(f"GGF TOTAL xsec 300_450: {totxsec_ggf_300_450}")
print(f"GGF TOTAL xsec 450_inf: {totxsec_ggf_450_inf}")

print("----------UNCERTAINTIES----------")
print(f"scl_tot_ggf:")
for name, array in scl_tot_ggf.items():
    print(f"{name}: {array}")
print(f"scl_reco_ggf_200_300:")
for name, array in scl_reco_ggf_200_300.items():
    print(f"{name}: {array}")
print(f"scl_reco_ggf_300_450:")
for name, array in scl_reco_ggf_300_450.items():
    print(f"{name}: {array}")
print(f"scl_reco_ggf_450_inf:")
for name, array in scl_reco_ggf_450_inf.items():
    print(f"{name}: {array}")



