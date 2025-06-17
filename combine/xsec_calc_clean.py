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

homedir =  '/eos/uscms/store/user/fmokhtar/boostedhiggs/Jun13_hww_stxs_'
years = ['2016','2016APV','2017','2018']
lumi = [16809.96, 19492.72, 41476.02, 59816.23]
xsecs = {'vbf': 0.8082134, 'ggf1a': 0.10078092000000001, 'ggf1b': 4.498172726490241, 'ggf2a': 0.10078092000000001, 'ggf2b': 4.498172726490241, 'ggf3a': 0.10078092000000001, 'ggf3b': 4.498172726490241}
procs = ['vbf', 'ggf1a', 'ggf1b', 'ggf2a', 'ggf2b', 'ggf3a', 'ggf3b'] 
#ggf1: ggf_200_300 ggf2: ggf_300_450 ggf3: ggf_450_inf
# ggf a = GluGluHToWW_Pt-200ToInf_M-125_Rivet
# ggf b = GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8
vbf_dir = 'VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'
ggfa_dir = 'GluGluHToWW_Pt-200ToInf_M-125_Rivet'
ggfb_dir = 'GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8'

#sumgenweights
sgw_tot = {proc: [0, 0, 0, 0] for proc in procs}
sgw_pass = {proc: [0, 0, 0, 0] for proc in procs}
totxsec = {proc: 0.0 for proc in procs}
#pklsumgenweight = {proc: 0.0 for proc in procs} #for debugging
pklsumgenweight = {proc: [0, 0, 0, 0] for proc in procs} #for debugging

#parton shower
psweights = ['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown'] #stored as weight_ele_PSISRUp
#make nested dict for initialization [proc][isr/fsr/up/down][year]
psdict_template = {'PSISRUp': [0,0,0,0], 'PSISRDown': [0,0,0,0], 'PSFSRUp': [0,0,0,0], 'PSFSRDown': [0,0,0,0]}
ps_pass = {proc: copy.deepcopy(psdict_template) for proc in procs}
ps_reco = {proc: copy.deepcopy(psdict_template) for proc in procs}
pstot_template = {'PSISRUp': 0.0, 'PSISRDown': 0.0, 'PSFSRUp': 0.0, 'PSFSRDown': 0.0}
totsyst_ps = {proc: copy.deepcopy(pstot_template) for proc in procs}

#QCD scale
scnames = ['weight_scale0', 'weight_scale1', 'weight_scale3', 'weight_scale5', 'weight_scale7', 'weight_scale8', 'weight_scale4']
sysdict_template = {'Up': [0,0,0,0], 'Down': [0,0,0,0]}
sc_pass = {proc: copy.deepcopy(sysdict_template) for proc in procs}
sc_reco = {proc: copy.deepcopy(sysdict_template) for proc in procs}
systot_template = {'Up': 0.0, 'Down': 0.0}
totsyst_scale = {proc: copy.deepcopy(systot_template) for proc in procs}

def calcxsec(sumweightspass, sumweightstotal, lum, xs):
    calc = (sumweightspass / sumweightstotal ) * lum * xs
    #print(calc)
    return calc

def getprocpath(proc):
    proc_path = ''
    if proc=='vbf':
        proc_path = vbf_dir
    elif proc in ['ggf1a', 'ggf2a', 'ggf3a']:
        proc_path = ggfa_dir
    elif proc in ['ggf1b', 'ggf2b', 'ggf3b']:
        proc_path = ggfb_dir
    return proc_path

#gets df from file directory. proc=ggf or vbf endstr is lep + '.parquet' or pkl
def getfilelist(proc, year, endstr): #with no selections the lepton doesnt matter
    filelist = []
    proc_path = getprocpath(proc)
    base = homedir+year
    dirs = [os.path.join(base, proc_path, 'outfiles/')]
    
    for directory in dirs:
        for filename in os.listdir(directory):
            if filename.endswith(endstr):
                filelist.append(os.path.join(directory, filename))
    return filelist

#returns dataframe with selection
#sel gen reco or base
#proc vbf ggf1 ggf2 ggf3
def selectdf(df, sel, proc):
    newdf = df #default
    # Extract base proc and file variant (e.g. 'ggf1a' → 'ggf1', 'a')
    base_proc = proc[:4] if proc.startswith('ggf') else proc
    variant = proc[4:] if proc.startswith('ggf') and len(proc) > 4 else None

    # STXS_finecat category selection
    stxs_map = {
        'vbf': [21, 22, 23, 24],
        'ggf1': [1, 5],
        'ggf2': [2, 6],
        'ggf3': [3, 4, 7, 8]
    }
    base_mask = (df['STXS_finecat'] % 100).isin(stxs_map[base_proc])

    if sel == 'gen':
        newdf = df[base_mask]  
    elif sel == 'reco':
        mask = base_mask #gen selection
        if base_proc == 'vbf':
            #mask &= (df['mjj'] > 1000) & (df['deta'] > 3.5)
            mask &= (df['mjj'] > 1000) & (df['deta'] > 3.5) & (df['NumOtherJets'] >= 2)
        else: #ggf
            #mask &= (df['mjj'] < 1000) | (df['deta'] < 3.5)
            mask &= (df['mjj'] < 1000) | (df['deta'] < 3.5) | (df['NumOtherJets'] < 2)
            if base_proc == 'ggf1':
                mask &= (df['rec_higgs_pt'] > 250) & (df['rec_higgs_pt'] < 350)
            elif base_proc == 'ggf2':
                mask &= (df['rec_higgs_pt'] > 350) & (df['rec_higgs_pt'] < 500)
            elif base_proc == 'ggf3':
                mask &= (df['rec_higgs_pt'] > 500) & (df['rec_higgs_pt'] < 2500)
        # Apply extra cut based on variant, does nothing to vbf 
        if variant == 'a':
            mask &= df['fj_genH_pt'] < 200
        elif variant == 'b':
            mask &= df['fj_genH_pt'] >= 200
        newdf = df[mask]
    else:
        raise ValueError(f"Unknown proc: {proc}")
    return newdf

#compute total cross section
def compute_xsec(lep='mu', debug=False):
    weight = 'weight_'+lep+'_genweight'
    endstr = lep+'.parquet'
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            print(f"year: {year}, process: {proc} file: {procfull}")
            filelist = getfilelist(proc, year, endstr)
            for filepath in filelist:
                events = pd.read_parquet(filepath)
                events_pass = selectdf(events, 'gen', proc)
                totsum = events[weight].sum()
                passsum = events_pass[weight].sum()
                sgw_tot[proc][i] += float(totsum)
                sgw_pass[proc][i] += float(passsum)                

            if debug: #check that genweight agrees with sumgenweight
                filelist = getfilelist(proc, year, '.pkl')
                for filepath in filelist:
                    with open(filepath, 'rb') as pklfile:
                        data = pickle.load(pklfile)
                        pklsum = data[procfull][year]['sumgenweight']
                        pklsumgenweight[proc][i] += float(pklsum)
                               
        # compute total cross section for the process
        totxsec[proc] = calcxsec(sum(sgw_pass[proc]), sum(sgw_tot[proc]), sum(lumi), xsecs[proc]) 
        print('----------------------------------------')
        print(f'sumgenweight_pass: {sum(sgw_pass[proc])}')
        print(f'sumgenweight_total: {sum(sgw_tot[proc])}')
        print(f'xsec: {totxsec[proc]}')

        if debug:
            print(f"pickle sumgenweight: {sum(pklsumgenweight[proc])}")
            print("Pickle total per year:")
            print(list(zip(years,pklsumgenweight[proc])))
            print("Total per year:")
            print(list(zip(years,sgw_tot[proc])))
        print('----------------------------------------')

#parton shower systematics
def compute_ps_systs(lep='mu'):
    preamble = 'weight_'+lep+'_'
    endstr = lep+'.parquet'
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            print(f"year: {year}, process: {proc} file: {procfull}")
            filelist = getfilelist(proc, year, endstr)
            for filepath in filelist:
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
def compute_scale_systs(lep='mu'):
    endstr = lep+'.parquet'
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            tot_scsums_pass = {name: 0.0 for name in scnames} #dictionary to hold the total sums per weight per year 
            tot_scsums_reco = {name: 0.0 for name in scnames}
            print(f"year: {year}, process: {proc} file: {procfull}")
            filelist = getfilelist(proc, year, endstr)
            for filepath in filelist:
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
compute_xsec(debug=False)
#compute_ps_systs()
#compute_scale_systs()

                    
