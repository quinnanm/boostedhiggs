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
pdfnames = [f"weight_pdf{i}" for i in range(100)] #all pdf weights from 0 to 100
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
    #slore nominal weights
    nominal_gen_totals  = {proc: [] for proc in procs}
    nominal_reco_totals = {proc: [] for proc in procs}
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            print(f"year: {year}, process: {proc} file: {procfull}")
            nom_pass_year = 0.0
            nom_reco_year = 0.0
            filelist = getfilelist(proc, year, endstr)
            for filepath in filelist:
                events = pd.read_parquet(filepath)
                events_gen = selectdf(events, 'gen', proc)
                events_reco = selectdf(events, 'reco', proc)
                #nominal weights
                nom_pass_year += float((events_gen[f"weight_{lep}"] * xsecs[proc]).sum())
                nom_reco_year += float((events_reco[f"weight_{lep}"] * xsecs[proc]).sum())
                for name in psweights: #['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown']
                    weight = preamble + name #weight_ele_PSISRUp
                    gensum = events_gen[weight].sum()
                    recosum = events_reco[weight].sum()
                    ps_pass[proc][name][i] += float(gensum) #[proc][isr/fsr/up/down][year]
                    ps_reco[proc][name][i] += float(recosum)
            nominal_gen_totals [proc].append(nom_pass_year)
            nominal_reco_totals[proc].append(nom_reco_year)
        
        #compute total systematic uncertainty for the process by summing weights per process over all years
        #nominal acceptance
        acc_nom = sum(nominal_reco_totals[proc]) / sum(nominal_gen_totals[proc])

        print('----------------------------------------')
        for name in psweights: #['PSISRUp', 'PSISRDown', 'PSFSRUp', 'PSFSRDown']
            raw_acc = sum(ps_reco[proc][name]) / sum(ps_pass[proc][name])
            #store the shift relative to nominal acceptance
            if name.endswith('Up'):
                totsyst_ps[proc][name] = raw_acc - acc_nom
            elif name.endswith('Down'):
                totsyst_ps[proc][name] = acc_nom - raw_acc
            print(f'ps syst name: {name}---------')
            print(f'ps_reco: {sum(ps_reco[proc][name])}')
            print(f'ps_pass: {sum(ps_pass[proc][name])}')
            print(f'nominal acc: {acc_nom:.6f}')
            print(f"{name}: raw acc = {raw_acc:.6f}, shift = {totsyst_ps[proc][name]:.10f}")
        print(f"tot ps up: {math.sqrt((totsyst_ps[proc]['PSISRUp'])**2 + (totsyst_ps[proc]['PSFSRUp'])**2)}")
        print(f"tot ps down: {math.sqrt((totsyst_ps[proc]['PSISRDown'])**2 + (totsyst_ps[proc]['PSFSRDown'])**2)}")

        print('----------------------------------------')


#qcd scale systematics
#up/down variations computed per year        
def compute_scale_systs(lep='mu'):
    endstr = lep+'.parquet'
    for proc in procs:
        # lists to collect nominal sums each year
        nominal_gen_totals  = []
        nominal_reco_totals = []
        #collect the central (scale4) yields per year
        central_pass_list = []
        central_reco_list = []
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
            # #first flatten the per event df arrays
            for weight in scnames:
                scweights_gen[weight] = pd.concat(scweights_gen[weight]).values
                scweights_reco[weight] = pd.concat(scweights_reco[weight]).values
            nominal_gen = pd.concat(nominal_gen).values
            nominal_reco = pd.concat(nominal_reco).values
            
            print(f"[{year}][{proc}] Nominal GEN total weight:   {nominal_gen.sum():.6g}")
            print(f"[{year}][{proc}] Nominal RECO total weight:  {nominal_reco.sum():.6g}")
            print(f"[{year}][{proc}] sumgenweight (GEN):       {sgw_pass [proc][i]:.6g}")
            print(f"[{year}][{proc}] sumgenweight (RECO):      {sgw_reco[proc][i]:.6g}")

            # stash the central (nominal) GEN/RECO sums for this year
            nominal_gen_totals.append(nominal_gen.sum())
            nominal_reco_totals.append(nominal_reco.sum())

            variation_yields_pass = {}
            variation_yields_reco = {}

            # #now to compute
            for weight in scnames:
                R_pass = tot_scsums_pass[weight]/sgw_pass[proc][i]
                R_reco = tot_scsums_reco[weight]/sgw_reco[proc][i]

                variation_yields_pass[weight] = (scweights_gen[weight] * nominal_gen / R_pass).sum()
                variation_yields_reco[weight] = (scweights_reco[weight] * nominal_reco / R_reco).sum()

            
            cen_gen  = variation_yields_pass['weight_scale4']
            cen_reco = variation_yields_reco['weight_scale4']
            print(f" central acc via scale4 = {cen_reco/cen_gen:.6f}  vs  nominal acc = {nominal_reco.sum()/nominal_gen.sum():.6f}")
            # stash the scale4 (“central”) yields for this year
            central_pass_list.append(cen_gen)
            central_reco_list.append(cen_reco)
           
            # ── DEBUG ──
            print(f"[{year}][{proc}] **Per-variation yields & acceptance**")
            for w in scnames:
                gen_yield = variation_yields_pass[w]
                reco_yield = variation_yields_reco[w]
                acc_var = reco_yield / gen_yield
                R_pass = tot_scsums_pass[w]  / sgw_pass [proc][i]
                R_reco = tot_scsums_reco[w]  / sgw_reco[proc][i]
                print(f"  {w:15s}  yield_gen={variation_yields_pass[w]:.3f}  "
                f"yield_reco={variation_yields_reco[w]:.3f}  "
                f"acc={variation_yields_reco[w]/variation_yields_pass[w]:.6f}")
                print()
            # ── END DEBUG ──

            variations = [w for w in scnames if w != 'weight_scale4']
            # now take the max/min over the TOTAL yields
            up_pass   = max(variation_yields_pass[w]   for w in variations)
            down_pass = min(variation_yields_pass[w]   for w in variations)
            up_reco   = max(variation_yields_reco[w]   for w in variations)
            down_reco = min(variation_yields_reco[w]   for w in variations)

            #save up and down scale weight sums
            sc_pass[proc]['Up'][i]   += up_pass
            sc_pass[proc]['Down'][i] += down_pass
            sc_reco[proc]['Up'][i]   += up_reco
            sc_reco[proc]['Down'][i] += down_reco

        #compute total systematic uncertainty for the process by summing weights per process over all years
        #combine the central sums into the nominal acceptance- using computed nominal weight
        total_nom_pass  = sum(nominal_gen_totals)
        total_nom_reco = sum(nominal_reco_totals)
        acc_nom         = total_nom_reco / total_nom_pass
        #combine the central sums into the nominal acceptance- using STORED central weight_scale4
        total_cen_pass  = sum(central_pass_list)
        total_cen_reco = sum(central_reco_list)
        acc_central    = total_cen_reco/total_cen_pass

        # compute the varied acceptances
        acc_up = sum(sc_reco [proc]['Up'])   / sum(sc_pass [proc]['Up'])
        acc_dn = sum(sc_reco [proc]['Down']) / sum(sc_pass [proc]['Down'])

        # store the shift relative to nominal acceptance 
        totsyst_scale[proc]['Up']   = acc_up - acc_nom
        totsyst_scale[proc]['Down'] = acc_nom - acc_dn
        # print only the nominal & varied acceptances
        print('----------------------------------------')
        print('----------RELATIVE TO NOMINAL WEIGHT----------------')
        #print(f"[{proc}] nominal acc = {acc_nom:.6f}, acc_up = {acc_up:.6f}, acc_dn = {acc_dn:.6f}")
        print(f"[{proc}] central acc (scale4) = {acc_central:.6f} | up = {acc_up:.6f} | dn = {acc_dn:.6f}")
        print(f"[{proc}] Final scale syst Up   = {totsyst_scale[proc]['Up']:.10f}")
        print(f"[{proc}] Final scale syst Down = {totsyst_scale[proc]['Down']:.10f}")
        print('----------------------------------------')

        # store the shift relative to central weight acceptance weight_scale4
        totsyst_scale[proc]['Up']   = acc_up      - acc_central
        totsyst_scale[proc]['Down'] = acc_central - acc_dn
        # print only the nominal & varied acceptances
        print('----------------------------------------')
        print('----------RELATIVE TO CENTRAL WEIGHT4----------------')
        #print(f"[{proc}] nominal acc = {acc_nom:.6f}, acc_up = {acc_up:.6f}, acc_dn = {acc_dn:.6f}")
        print(f"[{proc}] central acc (scale4) = {acc_central:.6f} | up = {acc_up:.6f} | dn = {acc_dn:.6f}")
        print(f"[{proc}] Final scale syst Up   = {totsyst_scale[proc]['Up']:.10f}")
        print(f"[{proc}] Final scale syst Down = {totsyst_scale[proc]['Down']:.10f}")
        print('----------------------------------------')


#pdf and alpha s
def compute_pdf_systs(lep='mu', typ='pdf'):
    #typ is pdf or als
    endstr = lep+'.parquet'
    #nominal weights
    nominal_gen_totals  = {proc: [] for proc in procs}
    nominal_reco_totals = {proc: [] for proc in procs}
    if typ == 'pdf':
        sysnames = pdfnames
    elif typ == 'als': 
        sysnames = alsnames
    for proc in procs:
        for i,year in enumerate(years):
            procfull = getprocpath(proc) #full name of sample
            nom_pass_year = 0.0 #for nominal weights
            nom_reco_year = 0.0
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
                wnom_gen = (events_gen[f"weight_{lep}"] * xsecs[proc]).sum()
                wnom_rec = (events_reco[f"weight_{lep}"] * xsecs[proc]).sum()
                nominal_gen.append(events_gen[f"weight_{lep}"]*xsecs[proc])
                nominal_reco.append(events_reco[f"weight_{lep}"]*xsecs[proc])
                nom_pass_year += float(wnom_gen)
                nom_reco_year += float(wnom_rec)
            nominal_gen_totals[proc].append(nom_pass_year)
            nominal_reco_totals[proc].append(nom_reco_year)
                     
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

            #now compute and save the up and down variations as hessian envelope
            if typ=='pdf':
                #this part is fishy- nick said should maybe not be taking the norm and then summing, do like a hessian or std instead?
                #saving it here becase it is the same as farouk has here: https://github.com/farakiko/boostedhiggs/blob/main/combine/make_templates.py#L144-L188
                # per‐event deviations
                # absunc_pdf_gen  = np.linalg.norm(allpdfweights_gen  - nominal_gen[:,None], axis=1)
                # absunc_pdf_reco = np.linalg.norm(allpdfweights_reco - nominal_reco[:,None], axis=1)

                # #fractional unc, capped at 100%
                # rel_unc_gen  = np.clip(absunc_pdf_gen  / nominal_gen, 0, 1)
                # rel_unc_reco = np.clip(absunc_pdf_reco / nominal_reco, 0, 1)

                # #per-event up/down weights
                # pdfup_gen   = nominal_gen  * (1 + rel_unc_gen)
                # pdfdown_gen = nominal_gen  * (1 - rel_unc_gen)
                # pdfup_reco  = nominal_reco * (1 + rel_unc_reco)
                # pdfdown_reco= nominal_reco * (1 - rel_unc_reco)

                # #store
                # pdf_pass[proc]['Up'][i]   += pdfup_gen.sum()
                # pdf_pass[proc]['Down'][i] += pdfdown_gen.sum()
                # pdf_reco[proc]['Up'][i]   += pdfup_reco.sum()
                # pdf_reco[proc]['Down'][i] += pdfdown_reco.sum()

                
                #alternative way to get up/down variations std that I used in prev analysis
                #could also do a hessian?
                nom_pass = nominal_gen.sum()
                nom_reco = nominal_reco.sum()

                #total yield
                yields_pass = np.array([(pdfweights_gen[w]  * nominal_gen  / (tot_pdfsums_pass[w]/sgw_pass[proc][i])).sum() for w in sysnames])
                yields_reco = np.array([(pdfweights_reco[w] * nominal_reco / (tot_pdfsums_reco[w]/sgw_reco[proc][i])).sum() for w in sysnames])

                #get the RMS (std-dev)
                sigma_pass = yields_pass.std(ddof=1)
                sigma_reco = yields_reco.std(ddof=1)

                #totals
                pdfup_pass_total   = nom_pass + sigma_pass
                pdfdown_pass_total = nom_pass - sigma_pass
                pdfup_reco_total   = nom_reco + sigma_reco
                pdfdown_reco_total = nom_reco - sigma_reco

                #store
                pdf_pass[proc]['Up'][i]   += pdfup_pass_total
                pdf_pass[proc]['Down'][i] += pdfdown_pass_total
                pdf_reco[proc]['Up'][i]   += pdfup_reco_total
                pdf_reco[proc]['Down'][i] += pdfdown_reco_total

            elif typ=='als': #compute as envelope instead
                variation_pass = {}
                variation_reco = {}
                for weight in sysnames:  # ['weight_pdf101','weight_pdf102']
                    R_pass = tot_pdfsums_pass[weight] / sgw_pass[proc][i]
                    R_reco = tot_pdfsums_reco[weight] / sgw_reco[proc][i]
                    variation_pass[weight] = (pdfweights_gen [weight] * nominal_gen  / R_pass).sum()
                    variation_reco[weight] = (pdfweights_reco[weight] * nominal_reco / R_reco).sum()

                #envelope on the TOTAL yields
                up_pass   = max(variation_pass[w] for w in sysnames)
                down_pass = min(variation_pass[w] for w in sysnames)
                up_reco   = max(variation_reco[w] for w in sysnames)
                down_reco = min(variation_reco[w] for w in sysnames)

                #store
                als_pass[proc]['Up']  [i] += up_pass
                als_pass[proc]['Down'][i] += down_pass
                als_reco[proc]['Up']  [i] += up_reco
                als_reco[proc]['Down'][i] += down_reco
            
        #compute total systematic uncertainty for the process by summing weights per process over all years
        #nominal
        acc_nom = sum(nominal_reco_totals[proc]) / sum(nominal_gen_totals[proc])
        print('----------------------------------------')
        for name in ['Up', 'Down']:
            if typ == 'pdf':
                raw_acc = sum(pdf_reco[proc][name]) / sum(pdf_pass[proc][name])
            elif typ=='als':
                raw_acc = sum(als_reco[proc][name]) / sum(als_pass[proc][name])

            #shift relative to nominal acceptance
            if name == 'Up':
                shift = raw_acc - acc_nom
            elif name == 'Down': 
                shift = acc_nom - raw_acc
            if typ == 'pdf':
                totsyst_pdf[proc][name] = shift
                print(f'pdf_reco {name}: {sum(pdf_reco[proc][name])}')
                print(f'pdf_pass {name}: {sum(pdf_pass[proc][name])}')
                print(f'nominal acc: {acc_nom:.6f}')
                print(f'PDF {name}: raw acc = {raw_acc:.6f}, shift = {shift:.10f}')
            elif typ=='als':
                totsyst_als[proc][name] = shift
                print(f'als_reco {name}: {sum(als_reco[proc][name])}')
                print(f'als_pass {name}: {sum(als_pass[proc][name])}')
                print(f'nominal acc: {acc_nom:.6f}')
                print(f'als  {name}: raw acc = {raw_acc:.6f}, shift = {shift:.10f}')
        print('----------------------------------------')


###### execute
compute_xsec(debug=False)
#compute_ps_systs()
#compute_scale_systs()
#compute_pdf_systs(typ='pdf')
#compute_pdf_systs(typ='als')

                    
