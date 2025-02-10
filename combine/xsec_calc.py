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


# Replace 'your_file.parquet' with the path to your Parquet file
years = ['2016','2016APV','2017','2018']
lumi = [16809.96, 19492.72, 41476.02, 59816.23]
tot_vbf = 0
totxsec_vbf = 0.0


#VBF #######################
for i,year in enumerate(years):

    # Initialize total event count
    total_events = 0
    weight_sum = 0
    
    directory = '/eos/uscms/store/user/fmokhtar/boostedhiggs/Oct16_hww_stxs_'+year+'/VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil/outfiles/'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('ele.parquet'):
            df = pd.read_parquet(filepath)
            num_events = len(df)
            total_events += num_events
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as pklfile:
                data = pickle.load(pklfile)
                sumgenweight = data['VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil'][year]['sumgenweight']
                #print(sumgenweight)
                weight_sum += float(sumgenweight)

    #calc xsec
    xsec_vbf = 0.0
    nev_vbf = 0
    #xsec from here https://github.com/farakiko/boostedhiggs/blob/main/fileset/xsec_pfnano.json
    if total_events>0:
        # xsec_vbf = (weight_sum/(lumi[i]*total_events))*0.8082134
        nev_vbf = (total_events*0.8082134*lumi[i])/weight_sum
        xsec_vbf = nev_vbf/lumi[i]
    else:
        print("WARNING: NO EVENTS")
                
    # Print the total number of events
    print(year)
    print('----------------------------------------')
    print(f"VBF Total number of events: {total_events}")
    print(f"VBF sumgenweight: {weight_sum}")
    print(f"lumi: {lumi[i]}")
    print("=======> nevt = total_events * xsec * lumi[i] / weight_sum <============")
    print(f"VBF nevt: {nev_vbf}")
    # print("=======> xsec = (sum of the weights)/Lumi*Nevts <============")
    print(f"VBF xsec: {xsec_vbf}")
    print('----------------------------------------')

    tot_vbf+=nev_vbf

totxsec_vbf = tot_vbf/sum(lumi)
print(f"VBF TOTAL xsec: {totxsec_vbf}")


#GGF ########################
# r_ggH_pt200_300
# r_ggH_pt300_450
# r_ggH_pt450_inf 

tot_ggf = 0
totxsec_ggf = 0.0


for i,year in enumerate(years):

    # Initialize total event count
    total_200_300 = 0
    total_300_450 = 0
    total_450_inf = 0
    weight_sum = 0
    
    directory = '/eos/uscms/store/user/fmokhtar/boostedhiggs/Sep23_hww_stxs_'+year+'/GluGluHToWW_Pt-200ToInf_M-125/outfiles/'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('ele.parquet'):
           # print(filename)
            df = pd.read_parquet(filepath)

            events_200_300 = df[(df['fj_genH_pt'] >= 200) & (df['fj_genH_pt'] < 300)]
            events_300_450 = df[(df['fj_genH_pt'] >= 300) & (df['fj_genH_pt'] < 450)]
            events_450_inf = df[(df['fj_genH_pt'] >= 450)]

            # Count the number of selected events
            total_200_300 += len(events_200_300)
            total_300_450 += len(events_300_450)
            total_450_inf += len(events_450_inf)

        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as pklfile:
                data = pickle.load(pklfile)
                # print(data)
                sumgenweight = data['GluGluHToWW_Pt-200ToInf_M-125'][year]['sumgenweight']
                #print(sumgenweight)
                weight_sum += float(sumgenweight)

    nev_ggf = 0
    #calculate the cross section
    #get fraction for each pt bin
    total = total_200_300 + total_300_450 + total_450_inf
    frac_200_300 = total_200_300/total
    frac_300_450 = total_300_450/total
    frac_450_inf = total_450_inf/total

    #xsec from here https://github.com/farakiko/boostedhiggs/blob/main/fileset/xsec_pfnano.json
    nev_ggf = (lumi[i]*total*0.10078092000000001)/weight_sum
    xsec_ggf = (weight_sum/(lumi[i]*total))*0.10078092000000001
    xsec_200_300 = xsec_ggf*frac_200_300
    xsec_300_450 = xsec_ggf*frac_300_450
    xsec_450_inf = xsec_ggf*frac_450_inf

    tot_ggf+=nev_ggf
    
    # Print the total number of events
    print(year)
    print('----------------------------------------')
    print(f"GGF Total number of events_200_300: {total_200_300}")
    print(f"GGF Total number of events_300_450: {total_300_450}")
    print(f"GGF Total number of events_450_inf: {total_450_inf}")
    print(f"GGF Total sumgenweight: {weight_sum}")
    print(f"total: {total}, frac_200_300: {frac_200_300}, frac_300_450: {frac_300_450}, frac_450_inf: {frac_450_inf}")
    print(f"lumi: {lumi[i]}")
    # print("=======> xsec = (sum of weights)/Lumi*Nevts <============")
    # print(f"xsec_ggf: {xsec_ggf}")
    # print(f"xsec_200_300: {xsec_200_300}")
    # print(f"xsec_300_450: {xsec_300_450}")
    # print(f"xsec_450_inf: {xsec_450_inf}")
    print("=======> nevt = total_events * xsec * lumi[i] / weight_sum <============")
    print(f"GGF nevt: {nev_ggf}")
    print(f"GGF xsec: {nev_ggf/lumi[i]}")
    print('----------------------------------------')

totxsec_ggf = tot_ggf/sum(lumi)
print(f"GGF TOTAL xsec: {totxsec_ggf}")
print(f"xsec_200_300: {totxsec_ggf*frac_200_300}")
print(f"xsec_300_450: {totxsec_ggf*frac_300_450}")
print(f"xsec_450_inf: {totxsec_ggf*frac_450_inf}")


    # column_names = df.columns.tolist()
    # # Print all column names
    # for col in column_names:
    #     print(col)

    # weight_200_300 = weight_sum*frac_200_300
    # weight_300_450 = weight_sum*frac_300_450
    # weight_450_inf = weight_sum*frac_450_inf

    # xsec_200_300 = weight_200_300/(lumi[i]*total_200_300)
    # xsec_300_450 = weight_300_450/(lumi[i]*total_300_450)
    # xsec_450_inf = weight_450_inf/(lumi[i]*total_450_inf)

