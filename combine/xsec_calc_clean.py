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
import math

homedir =  '/eos/uscms/store/user/fmokhtar/boostedhiggs/Jun13_hww_stxs_'
years = ['2016','2016APV','2017','2018']
lumi = [16809.96, 19492.72, 41476.02, 59816.23]
xsecs = {'vbf': 0.8082134, 'ggf1a': 0.10078092000000001, 'ggf1b': 4.498172726490241, 'ggf2a': 0.10078092000000001, 'ggf2b': 4.498172726490241, 'ggf3a': 0.10078092000000001, 'ggf3b': 4.498172726490241}
#procs = ['vbf', 'ggf1a', 'ggf1b', 'ggf2a', 'ggf2b', 'ggf3a', 'ggf3b'] #ggfb is zero
procs = ['vbf', 'ggf1a', 'ggf2a','ggf3a'] 
#ggf1: ggf_200_300 ggf2: ggf_300_450 ggf3: ggf_450_inf
# ggf a = GluGluHToWW_Pt-200ToInf_M-125_Rivet
# ggf b = GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8
vbf_dir = 'VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'
ggfa_dir = 'GluGluHToWW_Pt-200ToInf_M-125_Rivet'
ggfb_dir = 'GluGluHToWWToLNuQQ_M-125_TuneCP5_13TeV_powheg_jhugen751_pythia8'

#sumgenweights
sgw_tot = {proc: [0, 0, 0, 0] for proc in procs}
sgw_pass = {proc: [0, 0, 0, 0] for proc in procs}
sgw_reco = {proc: [0, 0, 0, 0] for proc in procs}
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

#pdf and alphas weights
pdfnames = [f"weight_pdf{i}" for i in range(100)] #all pdf weights from 0 to 102
alsnames = ['weight_pdf101', 'weight_pdf102'] #the last 2 pdf weights
pdf_pass = {proc: copy.deepcopy(sysdict_template) for proc in procs}
pdf_reco = {proc: copy.deepcopy(sysdict_template) for proc in procs}
totsyst_pdf = {proc: copy.deepcopy(systot_template) for proc in procs}
als_pass = {proc: copy.deepcopy(sysdict_template) for proc in procs}
als_reco = {proc: copy.deepcopy(sysdict_template) for proc in procs}
totsyst_als = {proc: copy.deepcopy(systot_template) for proc in procs}

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
    # Apply extra cut based on variant, does nothing to vbf 
    if variant == 'a':
        base_mask &= df['fj_genH_pt'] >= 200
    elif variant == 'b':
        base_mask &= df['fj_genH_pt'] < 200
    
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
        newdf = df[mask]
    
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
        print(f"tot ps up: {math.sqrt((totsyst_ps[proc]['PSISRUp'])**2 + (totsyst_ps[proc]['PSFSRUp'])**2)}")
        print(f"tot ps down: {math.sqrt((totsyst_ps[proc]['PSISRDown'])**2 + (totsyst_ps[proc]['PSFSRDown'])**2)}")

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
            scweights_gen = {name: [] for name in scnames} 
            scweights_reco = {name: [] for name in scnames} 
            nominal_gen = []
            nominal_reco = []
            print(f"year: {year}, process: {proc} file: {procfull}")
            filelist = getfilelist(proc, year, endstr)
            #for first loop around get the sums  
            for filepath in filelist:
                events = pd.read_parquet(filepath)
                events_gen = selectdf(events, 'gen', proc)
                events_reco = selectdf(events, 'reco', proc)
                #get sums of scale weights per year over all samples                
                #compute gen sumgenweight if not computed yet
                sgwname = 'weight_'+lep+'_genweight'
                if (sgw_pass[proc][i]==0):
                    passsum = events_gen[sgwname].sum()
                    sgw_pass[proc][i] += float(passsum)
                #compute reco sumgenweight
                if (sgw_reco[proc][i]==0):
                    recosum = events_reco[sgwname].sum()
                    sgw_reco[proc][i] += float(recosum)
                #get sums of scale weights per year over all samples 
                for weight in scnames:
                    #gen first
                    scsum_gen = events_gen[weight].sum()
                    tot_scsums_pass[weight] += float(scsum_gen)
                    #then reco
                    scsum_reco = events_reco[weight].sum()
                    tot_scsums_reco[weight] += float(scsum_reco)
                    #then save arrays of all weights in samples by appending the dataframes
                    scweights_gen[weight].append(events_gen[weight])
                    scweights_reco[weight].append(events_reco[weight])             
                #nominal arrays  #nominal = df[f"weight_{ch}"] * xsecweight
                nominal_gen.append(events_gen[f"weight_{lep}"]*xsecs[proc])
                nominal_reco.append(events_reco[f"weight_{lep}"]*xsecs[proc])
                
            
            #out of file loop, time to compute things looping over each scweight
            #first flatten the df arrays
            for weight in scnames:
                scweights_gen[weight] = pd.concat(scweights_gen[weight]).values
                scweights_reco[weight] = pd.concat(scweights_reco[weight]).values
            nominal_gen = pd.concat(nominal_gen).values
            nominal_reco = pd.concat(nominal_reco).values
            
            #now to compute
            allscweights_gen = []
            allscweights_reco = []
            central_gen = None
            central_reco = None
            for weight in scnames:
                R_pass = tot_scsums_pass[weight]/sgw_pass[proc][i]
                R_reco = tot_scsums_reco[weight]/sgw_reco[proc][i]
                wi_gen = scweights_gen[weight] * nominal_gen/R_pass 
                wi_reco = scweights_reco[weight] * nominal_reco/R_reco
                if weight == 'weight_scale4':
                    central_gen = wi_gen
                    central_reco = wi_reco
                else:
                    allscweights_gen.append(wi_gen)
                    allscweights_reco.append(wi_reco)
            
            allscweights_gen = np.swapaxes(np.array(allscweights_gen), 0, 1)
            allscweights_reco = np.swapaxes(np.array(allscweights_reco), 0, 1)
            scaleup_gen = nominal_gen * np.max(allscweights_gen,axis=1) / central_gen
            scaledown_gen = nominal_gen * np.min(allscweights_gen,axis=1) / central_gen
            scaleup_reco = nominal_reco * np.max(allscweights_reco,axis=1) / central_reco
            scaledown_reco = nominal_reco * np.min(allscweights_reco,axis=1) / central_reco
            #save up and down scale weight sums
            sc_pass[proc]['Up'][i] += scaleup_gen.sum()
            sc_pass[proc]['Down'][i] += scaledown_gen.sum()
            sc_reco[proc]['Up'][i] += scaleup_reco.sum()
            sc_reco[proc]['Down'][i] += scaledown_reco.sum()

        #compute total systematic uncertainty for the process by summing weights per process over all years
        print('----------------------------------------')
        for name in ['Up', 'Down']:
            totsyst_scale[proc][name] = sum(sc_reco[proc][name])/sum(sc_pass[proc][name])
            print(f'sc_reco {name}: {sum(sc_reco[proc][name])}')
            print(f'sc_pass {name}: {sum(sc_pass[proc][name])}')
            print(f'totsyst_scale {name}: {totsyst_scale[proc][name]}')
        print('----------------------------------------')

def compute_pdf_systs(lep='mu', typ='pdf'):
    #typ is pdf or als
    endstr = lep+'.parquet'
    if typ == 'pdf':
        sysnames = pdfnames
    elif typ == 'als': 
        sysnames = alsnames
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            #pdf
            tot_pdfsums_pass = {name: 0.0 for name in sysnames} #dictionary to hold the total sums per weight per year 
            tot_pdfsums_reco = {name: 0.0 for name in sysnames}
            pdfweights_gen = {name: [] for name in sysnames} 
            pdfweights_reco = {name: [] for name in sysnames} 
            nominal_gen = []
            nominal_reco = []
            print(f"year: {year}, process: {proc} file: {procfull}")
            filelist = getfilelist(proc, year, endstr)
            #for first loop around get the sums  
            for filepath in filelist:
                events = pd.read_parquet(filepath)
                events_gen = selectdf(events, 'gen', proc)
                events_reco = selectdf(events, 'reco', proc)
                #get sums of scale weights per year over all samples                
                #compute gen sumgenweight if not computed yet
                sgwname = 'weight_'+lep+'_genweight'
                if (sgw_pass[proc][i]==0):
                    passsum = events_gen[sgwname].sum()
                    sgw_pass[proc][i] += float(passsum)
                #compute reco sumgenweight
                if (sgw_reco[proc][i]==0):
                    recosum = events_reco[sgwname].sum()
                    sgw_reco[proc][i] += float(recosum)
                #get sums of scale weights per year over all samples 
                for weight in sysnames:
                    pdfsum_gen = events_gen[weight].sum()
                    pdfsum_reco = events_reco[weight].sum()
                    tot_pdfsums_pass[weight] += float(pdfsum_gen)                       
                    tot_pdfsums_reco[weight] += float(pdfsum_reco)
                    pdfweights_gen[weight].append(events_gen[weight])
                    pdfweights_reco[weight].append(events_reco[weight])          
                #nominal arrays  #nominal = df[f"weight_{ch}"] * xsecweight
                nominal_gen.append(events_gen[f"weight_{lep}"]*xsecs[proc])
                nominal_reco.append(events_reco[f"weight_{lep}"]*xsecs[proc])     
                     
            #out of file loop, time to compute things looping over each weight
            #first flatten the df arrays
            for weight in sysnames:                    
                pdfweights_gen[weight] = pd.concat(pdfweights_gen[weight]).values
                pdfweights_reco[weight] = pd.concat(pdfweights_reco[weight]).values
            nominal_gen = pd.concat(nominal_gen).values
            nominal_reco = pd.concat(nominal_reco).values

            #now to compute
            allpdfweights_gen = []
            allpdfweights_reco = []
            for weight in sysnames:
                R_pass = tot_pdfsums_pass[weight]/sgw_pass[proc][i]
                R_reco = tot_pdfsums_reco[weight]/sgw_reco[proc][i]
                wi_gen = pdfweights_gen[weight] * nominal_gen/R_pass 
                wi_reco = pdfweights_reco[weight] * nominal_reco/R_reco
                allpdfweights_gen.append(wi_gen)
                allpdfweights_reco.append(wi_reco)
            allpdfweights_gen = np.swapaxes(np.array(allpdfweights_gen), 0, 1)
            allpdfweights_reco = np.swapaxes(np.array(allpdfweights_reco), 0, 1)

            #now compute and save the up and down variations
            if typ=='pdf':
                absunc_pdf_gen = np.linalg.norm(allpdfweights_gen- nominal_gen[:, np.newaxis], axis=1)
                absunc_pdf_reco = np.linalg.norm(allpdfweights_reco- nominal_reco[:, np.newaxis], axis=1)
                #relunc_pdf_gen = np.clip(absunc_pdf_gen / nominal_gen, 0, 1)
                #relunc_pdf_reco = np.clip(absunc_pdf_reco / nominal_reco, 0, 1)
                pdfup_gen = nominal_gen + absunc_pdf_gen
                pdfdown_gen = nominal_gen - absunc_pdf_gen
                pdfup_reco = nominal_reco + absunc_pdf_reco
                pdfdown_reco = nominal_reco - absunc_pdf_reco
                #save up and down pdf weight sums
                pdf_pass[proc]['Up'][i] += pdfup_gen.sum()
                pdf_pass[proc]['Down'][i] += pdfdown_gen.sum()
                pdf_reco[proc]['Up'][i] += pdfup_reco.sum()
                pdf_reco[proc]['Down'][i] += pdfdown_reco.sum()
            elif typ=='als': #compute as envelope instead
                var_up_gen = allpdfweights_gen[:, 0]
                var_down_gen = allpdfweights_gen[:, 1]
                var_up_reco = allpdfweights_reco[:, 0]
                var_down_reco = allpdfweights_reco[:, 1]
                max_var_gen = np.maximum.reduce([var_up_gen, var_down_gen, nominal_gen])
                min_var_gen = np.minimum.reduce([var_up_gen, var_down_gen, nominal_gen])
                max_var_reco = np.maximum.reduce([var_up_reco, var_down_reco, nominal_reco])
                min_var_reco = np.minimum.reduce([var_up_reco, var_down_reco, nominal_reco])
                pdfup_gen = max_var_gen
                pdfdown_gen = min_var_gen
                pdfup_reco = max_var_reco
                pdfdown_reco = min_var_reco
                als_pass[proc]['Up'][i] += pdfup_gen.sum()
                als_pass[proc]['Down'][i] += pdfdown_gen.sum()
                als_reco[proc]['Up'][i] += pdfup_reco.sum()
                als_reco[proc]['Down'][i] += pdfdown_reco.sum()
            
        #compute total systematic uncertainty for the process by summing weights per process over all years
        print('----------------------------------------')
        for name in ['Up', 'Down']:
            if typ=='pdf':
                print('pdf')
                totsyst_pdf[proc][name] = sum(pdf_reco[proc][name])/sum(pdf_pass[proc][name])
                print(f'pdf_reco {name}: {sum(pdf_reco[proc][name])}')
                print(f'pdf_pass {name}: {sum(pdf_pass[proc][name])}')
                print(f'totsyst_pdf {name}: {totsyst_pdf[proc][name]}')
            elif typ=='als':
                print('als')
                totsyst_als[proc][name] = sum(als_reco[proc][name])/sum(als_pass[proc][name])
                print(f'als_reco {name}: {sum(als_reco[proc][name])}')
                print(f'als_pass {name}: {sum(als_pass[proc][name])}')
                print(f'totsyst_als {name}: {totsyst_als[proc][name]}')
        print('----------------------------------------')


###### execute
#compute_xsec(debug=False)
#compute_ps_systs()
#compute_scale_systs()
compute_pdf_systs(typ='pdf')
compute_pdf_systs(typ='als')

                    
