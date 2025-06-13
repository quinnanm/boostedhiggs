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
import copy

homedir =  '/eos/uscms/store/user/fmokhtar/boostedhiggs/Jun9_hww_stxs_'
years = ['2016','2016APV','2017','2018']
lumi = [16809.96, 19492.72, 41476.02, 59816.23]
xsecs = {'vbf': 0.8082134, 'ggf1': 0.10078092000000001, 'ggf2': 0.10078092000000001, 'ggf3': 0.10078092000000001}
procs = ['vbf', 'ggf1', 'ggf2', 'ggf3'] #ggf1: ggf_200_300 ggf2: ggf_300_450 ggf3: ggf_450_inf

#sumgenweights
sgw_tot = {'vbf': [0,0,0,0], 'ggf1': [0,0,0,0], 'ggf2': [0,0,0,0], 'ggf3': [0,0,0,0]}
sgw_pass = {'vbf': [0,0,0,0], 'ggf1': [0,0,0,0], 'ggf2': [0,0,0,0], 'ggf3': [0,0,0,0]}
totxsec = {'vbf': 0.0, 'ggf1': 0.0, 'ggf2': 0.0, 'ggf3': 0.0}
pklsumgenweight = {'vbf': 0.0, 'ggf1': 0.0, 'ggf2': 0.0, 'ggf3': 0.0} #for debugging

#parton shower
psweights = ['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown'] #stored as weight_ele_PSISRUp
#make nested dict for initialization [proc][isr/fsr/up/down][year]
psdict_template = {'PSISRUp': [0,0,0,0], 'PSISRDown': [0,0,0,0], 'PSFSRUp': [0,0,0,0], 'PSFSRDown': [0,0,0,0]}
ps_pass = {proc: copy.deepcopy(psdict_template) for proc in ['vbf', 'ggf1', 'ggf2', 'ggf3']}
ps_reco = {proc: copy.deepcopy(psdict_template) for proc in ['vbf', 'ggf1', 'ggf2', 'ggf3']}
totsyst_ps = {'vbf': {'PSISRUp':0.0, 'PSISRDown':0.0, 'PSFSRUp':0.0, 'PSFSRDown':0.0},
              'ggf1': {'PSISRUp':0.0, 'PSISRDown':0.0, 'PSFSRUp':0.0, 'PSFSRDown':0.0},
              'ggf2': {'PSISRUp':0.0, 'PSISRDown':0.0, 'PSFSRUp':0.0, 'PSFSRDown':0.0},
              'ggf3': {'PSISRUp':0.0, 'PSISRDown':0.0, 'PSFSRUp':0.0, 'PSFSRDown':0.0}}

#QCD scale
scnames = ['weight_scale0', 'weight_scale1', 'weight_scale3', 'weight_scale5', 'weight_scale7', 'weight_scale8', 'weight_scale4']
sysdict_template = {'Up': [0,0,0,0], 'Down': [0,0,0,0]}
sc_pass = {proc: copy.deepcopy(sysdict_template) for proc in ['vbf', 'ggf1', 'ggf2', 'ggf3']}
sc_reco = {proc: copy.deepcopy(sysdict_template) for proc in ['vbf', 'ggf1', 'ggf2', 'ggf3']}
totsyst_scale = {'vbf': {'Up':0.0, 'Down':0.0},
                 'ggf1': {'Up':0.0, 'Down':0.0},
                 'ggf2': {'Up':0.0, 'Down':0.0},
                 'ggf3': {'Up':0.0, 'Down':0.0}} 

def calcxsec(sumweightspass, sumweightstotal, lum, xs):
    calc = (sumweightspass / sumweightstotal ) * lum * xs
    #print(calc)
    return calc

#gets df from file directory. proc=ggf or vbf
def getdir(proc, year): #with no selections the lepton doesnt matter
    if 'vbf' in proc:
        procstr = '/VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet/outfiles/'
    elif 'ggf' in proc:
        procstr = '/GluGluHToWW_Pt-200ToInf_M-125_Rivet/outfiles/'
    directory = homedir + year + procstr
    return directory

#returns dataframe with selection
#sel gen reco or base
#proc vbf ggf1 ggf2 ggf3
def selectdf(df, sel, proc):
    newdf = df #default
    if proc=='vbf':
        if sel=='gen':
            newdf = df[ (df['STXS_finecat'] % 100).isin([21, 22, 23, 24]) ]
        elif sel=='reco': #gen+reco selections
            newdf = df[ (df['STXS_finecat'] % 100).isin([21, 22, 23, 24]) &
                        #(df['mjj'] > 1000) & (df['deta'] > 3.5) & (df['NumOtherJets'] >= 2) ]
                        (df['mjj'] > 1000) & (df['deta'] > 3.5) ]
    elif proc=='ggf1': #ggf_200_300
        if sel=='gen':
            newdf = df[ (df['STXS_finecat'] % 100).isin([1, 5]) ]
        elif sel=='reco':
            newdf = df[ (df['STXS_finecat'] % 100).isin([1, 5]) &
                        #((df['mjj'] < 1000) | (df['deta'] < 3.5) | (df['NumOtherJets'] < 2)) &
                        ((df['mjj'] < 1000) | (df['deta'] < 3.5)) &
                        (df['rec_higgs_pt'] > 250) &
                        (df['rec_higgs_pt'] < 350) ]
    elif proc=='ggf2': #ggf_300_450
        if sel=='gen':
            newdf = df[ (df['STXS_finecat'] % 100).isin([2, 6]) ]
        elif sel=='reco': #gen+reco selections
            newdf = df[ (df['STXS_finecat'] % 100).isin([2, 6]) &
                        #((df['mjj'] < 1000) | (df['deta'] < 3.5) | (df['NumOtherJets'] < 2)) &
                        ((df['mjj'] < 1000) | (df['deta'] < 3.5)) &
                        (df['rec_higgs_pt'] > 350) &
                        (df['rec_higgs_pt'] < 500) ]
    elif proc=='ggf3': #ggf_450_inf
        if sel=='gen':
            newdf = df[ (df['STXS_finecat'] % 100).isin([3, 4, 7, 8]) ]
        elif sel=='reco': #gen+reco selections
            newdf = df[ (df['STXS_finecat'] % 100).isin([3, 4, 7, 8]) &
                        #((df['mjj'] < 1000) | (df['deta'] < 3.5) | (df['NumOtherJets'] < 2)) &
                        ((df['mjj'] < 1000) | (df['deta'] < 3.5)) &
                        (df['rec_higgs_pt'] > 500) &
                        (df['rec_higgs_pt'] < 2500) ]
    else:
        raise ValueError(f"Unknown proc: {proc}")
    return newdf


def compute_xsec(lep='mu', debug=False):
    weight = 'weight_'+lep+'_genweight'
    for proc in procs:
        for i,year in enumerate(years):
            print(f"year: {year}, process: {proc}")
            directory = getdir(proc, year)
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if filename.endswith(lep+'.parquet'):
                    events = pd.read_parquet(filepath)
                    events_pass = selectdf(events, 'gen', proc)
                    totsum = events[weight].sum()
                    passsum = events_pass[weight].sum()
                    sgw_tot[proc][i] += float(totsum)
                    sgw_pass[proc][i] += float(passsum)                

                elif debug and filename.endswith('.pkl'):
                    #check that genweight agrees with sumgenweight
                    with open(filepath, 'rb') as pklfile:
                        data = pickle.load(pklfile)
                        procstr = 'GluGluHToWW_Pt-200ToInf_M-125_Rivet'
                        if 'vbf' in proc:
                            procstr = 'VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'
                        pklsumgenweight[proc] += data[procstr][year]['sumgenweight']                
                        
        # compute total cross section for the process
        totxsec[proc] = calcxsec(sum(sgw_pass[proc]), sum(sgw_tot[proc]), sum(lumi), xsecs[proc]) 
        print('----------------------------------------')
        print(f'sumgenweight_pass: {sum(sgw_pass[proc])}')
        print(f'sumgenweight_total: {sum(sgw_tot[proc])}')
        print(f'xsec: {totxsec[proc]}')
        if debug:
            print(f"pickle sumgenweight: {pklsumgenweight[proc]}")
        print('----------------------------------------')

#parton shower systematics
def compute_ps_systs(lep='mu'):
    preamble = 'weight_'+lep+'_'
    for proc in procs:
        for i,year in enumerate(years):
            print(f"year: {year}, process: {proc}")
            directory = getdir(proc, year)
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if filename.endswith(lep+'.parquet'):
                    events = pd.read_parquet(filepath)
                    events_gen = selectdf(events, 'gen', proc)
                    events_reco = selectdf(events, 'reco', proc)
                    for name in psweights: #['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown']
                        weight = preamble + name #weight_ele_PSISRUp
                        gensum = events_gen[weight].sum()
                        recosum = events_reco[weight].sum()
                        ps_pass[proc][name][i] += float(gensum) #[proc][isr/fsr/up/down][year]
                        ps_reco[proc][name][i] += float(recosum)
        #compute total systematic uncertainty for the process by summing weights per process over all years
        print('----------------------------------------')
        for name in psweights: #['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown']
            totsyst_ps[proc][name] = sum(ps_reco[proc][name])/sum(ps_pass[proc][name])
            print(f'ps syst name: {name}---------')
            print(f'ps_reco: {sum(ps_reco[proc][name])}')
            print(f'ps_pass: {sum(ps_pass[proc][name])}')
            print(f'totsyst_ps: {totsyst_ps[proc][name]}')
        print('----------------------------------------')

#qcd scale systematics
#up/down variations computed per year        
def compute_scale_systs(lep='mu')
    for proc in procs:
        for i,year in enumerate(years):
            tot_scsums_pass = {name: 0.0 for name in scnames} #dictionary to hold the total sums per weight per year 
            tot_scsums_reco = {name: 0.0 for name in scnames}
            print(f"year: {year}, process: {proc}")
            directory = getdir(proc, year)
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if filename.endswith(lep+'.parquet'):
                    events = pd.read_parquet(filepath)
                    events_gen = selectdf(events, 'gen', proc)
                    events_reco = selectdf(events, 'reco', proc)
                    sels = ['pass', 'reco']
                    #get sums of scale weights per year
                    for sel in sels:
                        for weight in scnames:
                            if sel == 'pass':
                                scsum = events_gen[weight].sum()
                                tot_scsums_pass[weight] += float(scsum)
                            elif sel == 'reco':
                                scsum = events_reco[weight].sum()
                                tot_scsums_reco[weight] += float(scsum)
                            
            

###### execute
#compute_xsec(debug=True)
#compute_ps_systs()
compute_scale_systs()

                    
