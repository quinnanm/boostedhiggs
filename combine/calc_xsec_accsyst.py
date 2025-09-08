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
lumi = {'2016':16809.96, '2016APV':19492.72, '2017':41476.02, '2018':59816.23}
#xsecs sans BR
#ggf gen xsec found here: https://github.com/farakiko/boostedhiggs/blob/main/fileset/xsec.py#L45
#vbf gen xsec found here: https://github.com/farakiko/boostedhiggs/blob/main/fileset/xsec.py#L70
genxsecs = {'vbf': 3.782, 'ggf1':  0.4716, 'ggf2':  0.4716, 'ggf3':  0.4716}
procs = ['vbf', 'ggf1', 'ggf2','ggf3'] 
#ggf1: ggf_200_300 ggf2: ggf_300_450 ggf3: ggf_450_inf
vbf_dir = 'VBFHToWWToAny_M-125_TuneCP5_withDipoleRecoil_Rivet'
ggf_dir = 'GluGluHToWW_Pt-200ToInf_M-125_Rivet'


def getprocpath(proc):
    proc_path = ''
    if proc=='vbf':
        proc_path = vbf_dir
    elif proc in ['ggf1', 'ggf2', 'ggf3']:
        proc_path = ggf_dir
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
    # STXS_finecat category selection
    stxs_map = {
        'vbf': [21, 22, 23, 24],
        'ggf1': [1, 5],
        'ggf2': [2, 6],
        'ggf3': [3, 4, 7, 8]
    }
    base_mask = (df['STXS_finecat'] % 100).isin(stxs_map[base_proc])
    # Apply extra cut to ggf, does nothing to vbf 
    if proc != 'vbf':
        base_mask &= df['fj_genH_pt'] >= 200
    
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


#compute nominal gen-bin cross section
def compute_xsec_gen(lep='mu', debug=False):
    """
    Compute nominal gen-bin cross sections:
        sigma_gen(B) = (S_gen / S_tot) * xs
    and cross-check S_tot against pickle 'sumgenweight'.
    """
    weight_col = f'weight_{lep}_genweight'
    endstr = f'{lep}.parquet'
    # Sums from parquet (nominal generator weights)
    S_tot  = {year: {proc: 0.0 for proc in procs} for year in years}
    S_gen  = {year: {proc: 0.0 for proc in procs} for year in years}
    # Cross-check from pickle files (total sumgenweight)
    P_tot  = {year: {proc: 0.0 for proc in procs} for year in years}
    # Results
    sigma_by_year = {year: {proc: 0.0 for proc in procs} for year in years}
    sigma_all     = {proc: 0.0 for proc in procs}

    # Only columns needed for *gen* selection
    needed_cols = ['STXS_finecat', 'fj_genH_pt', weight_col]

    for year in years:
        for proc in procs:
            procfull = getprocpath(proc)
            # ---- Parquet sums ----
            for fp in getfilelist(proc, year, endstr):
                df = pd.read_parquet(fp, columns=needed_cols)
                # total (no cuts)
                S_tot[year][proc] += df[weight_col].sum()
                # gen bin (STXS) cut
                df_gen = selectdf(df, 'gen', proc)
                S_gen[year][proc] += df_gen[weight_col].sum()

            # ---- Pickle cross-check ----
            if debug:
                for fp in getfilelist(proc, year, '.pkl'):
                    try:
                        with open(fp, 'rb') as pklfile:
                            data = pickle.load(pklfile)
                        # expected layout: data[procfull][year]['sumgenweight']
                        pklsum = float(data[procfull][year]['sumgenweight'])
                        P_tot[year][proc] += pklsum
                    except Exception as e:
                        print(f"[WARN] Could not read sumgenweight from {fp}: {e}")

            # σ_gen(B) = (S_gen / S_tot) * xs  (per year)
            denom = S_tot[year][proc]
            sigma_by_year[year][proc] = (S_gen[year][proc] / denom) * genxsecs[proc] if denom else 0.0

            if debug:
                frac = (S_gen[year][proc] / denom) if denom else 0.0
                print(f"[{year} {proc}] S_gen={S_gen[year][proc]:.6e}  "
                      f"S_tot={denom:.6e}  frac={frac:.6f}  "
                      f"sigma_gen={sigma_by_year[year][proc]:.6g} pb")

    # ---- Combine all years from sums ----
    for proc in procs:
        Sg = sum(S_gen[y][proc] for y in years)
        St = sum(S_tot[y][proc] for y in years)
        sigma_all[proc] = (Sg / St) * genxsecs[proc] if St else 0.0

        if debug:
            print('----------------------------------------')
            # Pickle totals (cross-check)
            pkl_tot_proc = sum(P_tot[y][proc] for y in years)
            print(f"[{proc}] pickle sumgenweight (ALL YEARS): {pkl_tot_proc:.6e}")
            print("Pickle total per year:")
            print(list(zip(years, [P_tot[y][proc] for y in years])))
            print("Parquet totals per year (S_tot):")
            print(list(zip(years, [S_tot[y][proc] for y in years])))

            parquet_tot_proc = sum(S_tot[y][proc] for y in years)
            if parquet_tot_proc != 0:
                reldiff = (pkl_tot_proc - parquet_tot_proc) / parquet_tot_proc
                print(f"Relative diff (pkl - parquet)/parquet = {reldiff:.3e}")
            print(f"[{proc}] overall σ_gen = {sigma_all[proc]:.6f} pb")
            print('----------------------------------------')

    return sigma_by_year, sigma_all, S_gen, S_tot, P_tot


#compute ps uncertainties
def compute_ps_acceptance(lep='mu', debug=False):
    """
    Parton-shower ISR/FSR relative acceptances (PS weights are ABSOLUTE).
      A_nom  = sum(base)_reco / sum(base)_gen
      A_var  = sum(base*PSvar)_reco / sum(base*PSvar)_gen
      rel    = A_var / A_nom
    Returns:
      per_year[year][proc] and combined[proc] with:
        {'A_nom': float, 'ISR': {'up','down'}, 'FSR': {'up','down'}}
    """
    base_col = f'weight_{lep}_genweight'
    ps_cols = {
        'ISRUp':   f'weight_{lep}_PSISRUp',
        'ISRDown': f'weight_{lep}_PSISRDown',
        'FSRUp':   f'weight_{lep}_PSFSRUp',
        'FSRDown': f'weight_{lep}_PSFSRDown'}
    variants = ['nom', 'ISRUp', 'ISRDown', 'FSRUp', 'FSRDown']
    #prep dictionaries
    # sums: S[year][proc][variant]['gen'|'reco'] -> float
    S = {year: {proc: {v: {'gen': 0.0, 'reco': 0.0} for v in variants}
                for proc in procs}
         for year in years}

    needed_cols = [
        'STXS_finecat','fj_genH_pt',                  # gen selection
        'rec_higgs_pt','mjj','deta','NumOtherJets',   # reco selection
        base_col, *ps_cols.values() ]
    
    #loop and fill
    for year in years:
        for proc in procs:
            for fp in getfilelist(proc, year, f'{lep}.parquet'):
                # read, tolerating missing columns
                try:
                    df = pd.read_parquet(fp, columns=needed_cols)
                except Exception:
                    df = pd.read_parquet(fp)
                if base_col not in df:
                    raise RuntimeError(f"{base_col} missing in {fp}")
                for c in ps_cols.values():
                    if c not in df:
                        df[c] = df[base_col]  # absolute convention: nominal = base weight
                        #df[c] = 1.0  # multiplicative factor weights = no change

                # selections
                df_gen  = selectdf(df,  'gen',  proc)
                df_reco = selectdf(df, 'reco',  proc)
                # nominal sums (factor = 1)
                S[year][proc]['nom']['gen']  += df_gen[base_col].sum()
                S[year][proc]['nom']['reco'] += df_reco[base_col].sum()
                # variations: multiply base by factor
                #note: absolute not multiplicative weights
                for tag, col in ps_cols.items():
                    S[year][proc][tag]['gen']  += df_gen[col].sum()
                    S[year][proc][tag]['reco'] += df_reco[col].sum()
                #absolute weights, not multiplicative (not used)
                #for tag, col in ps_cols.items():
                    #S[year][proc][tag]['gen']  += (df_gen[base_col]  * df_gen[col]).sum()
                    #S[year][proc][tag]['reco'] += (df_reco[base_col] * df_reco[col]).sum()

    def finalize(Syproc):
        eps = 1e-12
        gen_nom  = Syproc['nom']['gen']
        reco_nom = Syproc['nom']['reco']
        A_nom = (reco_nom / gen_nom) if abs(gen_nom) > eps else 0.0

        out = {'A_nom': A_nom, 'ISR': {'up': 1.0, 'down': 1.0}, 'FSR': {'up': 1.0, 'down': 1.0}}
        for tag, (group, ud) in {
            'ISRUp': ('ISR','up'), 'ISRDown': ('ISR','down'),
            'FSRUp': ('FSR','up'), 'FSRDown': ('FSR','down')
        }.items():
            g = Syproc[tag]['gen']; r = Syproc[tag]['reco']
            A_v = (r / g) if abs(g) > eps else A_nom
            out[group][ud] = (A_v / A_nom) if A_nom else 1.0
        return out

    # per-year results
    per_year = {year: {proc: finalize(S[year][proc]) for proc in procs} for year in years}
    # combined across years (aggregate sums first, then ratio)
    combined = {}
    for proc in procs:
        agg = {v: {'gen': 0.0, 'reco': 0.0} for v in variants}
        for year in years:
            for v in variants:
                agg[v]['gen']  += S[year][proc][v]['gen']
                agg[v]['reco'] += S[year][proc][v]['reco']
        combined[proc] = finalize(agg)

    if debug:
        for proc in procs:
            print('---', proc, '---')
            print('combined:', combined[proc])

    return per_year, combined



#compute scale uncertainties
def compute_scale_acceptance(lep='mu', debug=False):
    """
    QCD scale acceptance (multiplicative convention):
      A_nom  = sum(base * scale4)_reco / sum(base * scale4)_gen  (scale4 is nominal factor ~1)
      A_k    = sum(base * scale{k})_reco / sum(base * scale{k})_gen for k in {0,1,3,5,7,8}
      rel_k  = A_k / A_nom
      UP/DOWN = envelope over k in {0,1,3,5,7,8}

    Returns:
      per_year[year][proc] = {'A_nom': float, 'relatives': {0:...,1:...,3:...,5:...,7:...,8:...},
                              'UP': float, 'DOWN': float}
      combined[proc]       = same, aggregated across all years first (more correct).
    """
    base_col   = f'weight_{lep}_genweight'
    scale_nom  = 'weight_scale4'
    scale_vars = ['weight_scale0','weight_scale1','weight_scale3',
                  'weight_scale5','weight_scale7','weight_scale8']
    variants   = ['nom'] + scale_vars
    needed_cols = [
        'STXS_finecat','fj_genH_pt',         # gen selection
        'rec_higgs_pt','mjj','deta','NumOtherJets',  # reco selection
        base_col, scale_nom, *scale_vars]
    #prep dictionaries
    # S[year][proc][variant]['gen'|'reco'] -> float
    S = {year: {proc: {v: {'gen': 0.0, 'reco': 0.0} for v in variants}
                for proc in procs}
         for year in years}
    
    #loop and fill
    for year in years:
        for proc in procs:
            for fp in getfilelist(proc, year, f'{lep}.parquet'):
                df = pd.read_parquet(fp)
                # ensure missing factor cols default to 1.0
                if base_col not in df:
                    raise RuntimeError(f"{base_col} missing in {fp}")
                if scale_nom not in df:
                    df[scale_nom] = 1.0
                for c in scale_vars:
                    if c not in df:
                        df[c] = 1.0
                df_gen  = selectdf(df,  'gen',  proc)
                df_reco = selectdf(df, 'reco',  proc)
                # nominal: base * scale4 (here scale4==1)
                S[year][proc]['nom']['gen']  += (df_gen[base_col]  * df_gen[scale_nom]).sum()
                S[year][proc]['nom']['reco'] += (df_reco[base_col] * df_reco[scale_nom]).sum()
                # variations: base * scale{k}
                for col in scale_vars:
                    S[year][proc][col]['gen']  += (df_gen[base_col]  * df_gen[col]).sum()
                    S[year][proc][col]['reco'] += (df_reco[base_col] * df_reco[col]).sum()

    def finalize(Syproc):
        gen_nom  = Syproc['nom']['gen']
        reco_nom = Syproc['nom']['reco']
        A_nom = (reco_nom / gen_nom) if gen_nom else 0.0
        relatives = {}
        for col in scale_vars:
            gen_v, reco_v = Syproc[col]['gen'], Syproc[col]['reco']
            A_v = (reco_v / gen_v) if gen_v else A_nom
            relatives[int(col[-1])] = (A_v / A_nom) if A_nom else 1.0  # map 0,1,3,5,7,8
        up  = max(relatives.values()) if relatives else 1.0
        dn  = min(relatives.values()) if relatives else 1.0
        return {'A_nom': A_nom, 'relatives': relatives, 'UP': up, 'DOWN': dn}
    per_year = {year: {proc: finalize(S[year][proc]) for proc in procs} for year in years}

    # Aggregate across years first (better than averaging ratios)
    combined = {}
    for proc in procs:
        agg = {v: {'gen': 0.0, 'reco': 0.0} for v in variants}
        for year in years:
            for v in variants:
                agg[v]['gen']  += S[year][proc][v]['gen']
                agg[v]['reco'] += S[year][proc][v]['reco']
        combined[proc] = finalize(agg)

    if debug:
        for proc in procs:
            print('---', proc, '---')
            print('combined:', combined[proc])
            #for year in years:
                #print(year, per_year[year][proc])

    return per_year, combined


#compute pdf and alphas uncertainties
def compute_pdf_alphas_acceptance(lep='mu', debug=False):
    """
    Acceptance-only systematics for PDF (replicas 0..100) and alpha_s (101/102),
    using multiplicative factors: varied weight = base * weight_pdf{i}.

    PDF:
      A_nom     = sum(base)_reco / sum(base)_gen
      A_rep[i]  = sum(base*pdf_i)_reco / sum(base*pdf_i)_gen
      rel_i     = A_rep[i] / A_nom
      δ_pdf     = sqrt( mean_i (rel_i - 1)^2 )    # replica RMS
      PDF_UP    = 1 + δ_pdf
      PDF_DOWN  = 1 - δ_pdf

    αs:
      rel_101   = A_(101)/A_nom, rel_102 = A_(102)/A_nom
      ALPHAS_UP = max(rel_101, rel_102)
      ALPHAS_DN = min(rel_101, rel_102)
    """
    pdf_indices=range(0, 101) #0-100 inclusive. 101 and 102 are alphas
    base_col = f'weight_{lep}_genweight'
    alpha_cols = {101: 'weight_pdf101', 102: 'weight_pdf102'}

    needed_base = [
        'STXS_finecat','fj_genH_pt',
        'rec_higgs_pt','mjj','deta','NumOtherJets',
        base_col
    ]

    per_year = {year: {proc: None for proc in procs} for year in years}
    combined = {}

    def accumulate(year, proc):
        S_nom = {'gen': 0.0, 'reco': 0.0}
        S_pdf = {i: {'gen': 0.0, 'reco': 0.0} for i in pdf_indices}
        S_aS  = {k: {'gen': 0.0, 'reco': 0.0} for k in alpha_cols}

        for fp in getfilelist(proc, year, f'{lep}.parquet'):
            df0 = pd.read_parquet(fp)
            if base_col not in df0.columns:
                raise RuntimeError(f"{base_col} missing in {fp}")

            pdf_cols_present = [f'weight_pdf{i}' for i in pdf_indices if f'weight_pdf{i}' in df0.columns]
            aS_cols_present  = [alpha_cols[k] for k in alpha_cols if alpha_cols[k] in df0.columns]
            cols = list(set(needed_base + pdf_cols_present + aS_cols_present))
            df = df0[cols].copy()

            df_gen  = selectdf(df,  'gen',  proc)
            df_reco = selectdf(df, 'reco',  proc)

            # nominal (base only)
            S_nom['gen']  += df_gen[base_col].sum()
            S_nom['reco'] += df_reco[base_col].sum()

            # PDF replicas (factor): base * weight_pdf{i}
            for col in pdf_cols_present:
                i = int(col.replace('weight_pdf',''))
                S_pdf[i]['gen']  += (df_gen[base_col]  * df_gen[col]).sum()
                S_pdf[i]['reco'] += (df_reco[base_col] * df_reco[col]).sum()

            # alpha_s (factor): base * weight_pdf101/102
            for k, col in alpha_cols.items():
                if col in df.columns:
                    S_aS[k]['gen']  += (df_gen[base_col]  * df_gen[col]).sum()
                    S_aS[k]['reco'] += (df_reco[base_col] * df_reco[col]).sum()

        # finalize
        eps = 1e-12 # avoid div by zero
        A_nom = (S_nom['reco'] / S_nom['gen']) if abs(S_nom['gen']) > eps else 0.0

        # PDF relatives + RMS
        rel_pdf = {}
        for i, sums in S_pdf.items():
            g, r = sums['gen'], sums['reco']
            if abs(g) > eps and A_nom:
                rel_pdf[i] = (r/g) / A_nom
            else:
                rel_pdf[i] = 1.0
        vals = [v for v in rel_pdf.values() if np.isfinite(v)]
        delta_pdf = float(np.sqrt(np.mean([(v - 1.0)**2 for v in vals]))) if vals else 0.0
        pdf_up, pdf_dn = 1.0 + delta_pdf, 1.0 - delta_pdf

        # αs envelope
        rel_as = {}
        for k, sums in S_aS.items():
            g, r = sums['gen'], sums['reco']
            rel_as[k] = (r/g)/A_nom if abs(g) > eps and A_nom else 1.0
        aS_up = max(rel_as.values()) if rel_as else 1.0
        aS_dn = min(rel_as.values()) if rel_as else 1.0

        out = {
            'A_nom': A_nom,
            'pdf':   {'relatives': rel_pdf, 'RMS': delta_pdf, 'UP': pdf_up, 'DOWN': pdf_dn},
            'alphas':{'relatives': rel_as,   'UP': aS_up,     'DOWN': aS_dn}}
        return out, S_nom, S_pdf, S_aS

    sums_year = {}
    for year in years:
        sums_year[year] = {}
        for proc in procs:
            res, S_nom, S_pdf, S_aS = accumulate(year, proc)
            per_year[year][proc] = res
            sums_year[year][proc] = (S_nom, S_pdf, S_aS)

    # combine across years by aggregating sums, then recomputing ratios
    for proc in procs:
        S_nom = {'gen': 0.0, 'reco': 0.0}
        S_pdf = {}
        S_aS  = {}
        for year in years:
            Sn, Sp, Sa = sums_year[year][proc]
            S_nom['gen']  += Sn['gen'];  S_nom['reco'] += Sn['reco']
            for i, sums in Sp.items():
                S_pdf.setdefault(i, {'gen':0.0,'reco':0.0})
                S_pdf[i]['gen']  += sums['gen']; S_pdf[i]['reco'] += sums['reco']
            for k, sums in Sa.items():
                S_aS.setdefault(k, {'gen':0.0,'reco':0.0})
                S_aS[k]['gen']   += sums['gen']; S_aS[k]['reco']  += sums['reco']

        eps = 1e-12 # avoid div by zero
        A_nom = (S_nom['reco'] / S_nom['gen']) if abs(S_nom['gen']) > eps else 0.0

        rel_pdf = {i: ((S_pdf[i]['reco']/S_pdf[i]['gen'])/A_nom if abs(S_pdf[i]['gen'])>eps and A_nom else 1.0)
                   for i in S_pdf}
        vals = [v for v in rel_pdf.values() if np.isfinite(v)]
        delta_pdf = float(np.sqrt(np.mean([(v - 1.0)**2 for v in vals]))) if vals else 0.0
        pdf_up, pdf_dn = 1.0 + delta_pdf, 1.0 - delta_pdf

        rel_as = {k: ((S_aS[k]['reco']/S_aS[k]['gen'])/A_nom if abs(S_aS[k]['gen'])>eps and A_nom else 1.0)
                  for k in S_aS}
        aS_up = max(rel_as.values()) if rel_as else 1.0
        aS_dn = min(rel_as.values()) if rel_as else 1.0

        combined[proc] = {
            'A_nom': A_nom,
            'pdf':   {'relatives': rel_pdf, 'RMS': delta_pdf, 'UP': pdf_up, 'DOWN': pdf_dn},
            'alphas':{'relatives': rel_as,   'UP': aS_up,     'DOWN': aS_dn}}

        if debug:
            print('---', proc, '---')
            print('combined:', combined[proc])

    return per_year, combined


def _bracket_to_deltas(rel_up, rel_down):
    """
    Turn two relative factors (could be both <1, both >1, or straddling 1)
    into one-sided deltas suitable for quadrature combination across sources.

    Returns (delta_up, delta_down) with:
      delta_up  = max(0, rel_up - 1, rel_down - 1)
      delta_down= max(0, 1 - rel_up, 1 - rel_down)
    """
    du = max(0.0, rel_up - 1.0,  rel_down - 1.0)
    dd = max(0.0, 1.0 - rel_up, 1.0 - rel_down)
    return du, dd

#for computing the final combined uncertainties
def build_acceptance_systematics(ps_combined, sc_combined, pdf_combined, procs):
    """
    Inputs (per process):
      ps_combined[proc] = {'A_nom':..., 'ISR':{'up','down'}, 'FSR':{'up','down'}}
      sc_combined[proc] = {'A_nom':..., 'UP':..., 'DOWN':...}               # QCD scale envelope
      pdf_combined[proc]= {'A_nom':..., 'pdf':{'RMS', 'UP','DOWN'}, 'alphas':{'UP','DOWN'}}

    Output:
      totals[proc] = {
        'components': {
          'isr':    {'up': δ_up, 'down': δ_dn},
          'fsr':    {'up': δ_up, 'down': δ_dn},
          'qcd':    {'up': δ_up, 'down': δ_dn},
          'pdf':    {'up': δ,    'down': δ},
          'alphas': {'up': δ_up, 'down': δ_dn},
        },
        'total': {'delta_up': Δ_up, 'delta_down': Δ_dn, 'up': 1+Δ_up, 'down': 1-Δ_dn}
      }
    """
    totals = {}
    for proc in procs:
        comps = {}

        # ISR / FSR (relative factors already, e.g. 0.987, 1.012)
        isr_up, isr_dn = ps_combined[proc]['ISR']['up'], ps_combined[proc]['ISR']['down']
        fsr_up, fsr_dn = ps_combined[proc]['FSR']['up'], ps_combined[proc]['FSR']['down']
        comps['isr']    = dict(zip(['up','down'], _bracket_to_deltas(isr_up, isr_dn)))
        comps['fsr']    = dict(zip(['up','down'], _bracket_to_deltas(fsr_up, fsr_dn)))

        # QCD scale envelope (UP/DOWN are relative factors)
        sc_up, sc_dn = sc_combined[proc]['UP'], sc_combined[proc]['DOWN']
        comps['qcd']    = dict(zip(['up','down'], _bracket_to_deltas(sc_up, sc_dn)))

        # PDF (RMS is symmetric)
        pdf_delta = pdf_combined[proc]['pdf']['RMS']
        comps['pdf']    = {'up': pdf_delta, 'down': pdf_delta}

        # alpha_s (envelope)
        a_up, a_dn = pdf_combined[proc]['alphas']['UP'], pdf_combined[proc]['alphas']['DOWN']
        comps['alphas'] = dict(zip(['up','down'], _bracket_to_deltas(a_up, a_dn)))

        # Sum in quadrature
        upvals   = [c['up']   for c in comps.values()]
        downvals = [c['down'] for c in comps.values()]
        D_up   = math.sqrt(sum(v*v for v in upvals))
        D_down = math.sqrt(sum(v*v for v in downvals))

        totals[proc] = {
            'components': comps,
            'total': {'delta_up': D_up, 'delta_down': D_down, 'up': 1.0 + D_up, 'down': 1.0 - D_down}
        }
    return totals

def print_acceptance_totals(totals, procs):
    for proc in procs:
        t = totals[proc]['total']
        print(f"{proc}: total acceptance κ_up/κ_down = {t['up']:.6f} / {t['down']:.6f} "
              f"(Δ_up={t['delta_up']:.6e}, Δ_down={t['delta_down']:.6e})")


###### execute
sigma_by_year, sigma_all, S_gen, S_tot, P_tot = compute_xsec_gen(debug=True)
ps_peryear, ps_combined = compute_ps_acceptance(debug=True)
sc_peryear, sc_combined = compute_scale_acceptance(debug=True)
pdf_peryear, pdf_combined = compute_pdf_alphas_acceptance(debug=True)
totals = build_acceptance_systematics(ps_combined, sc_combined, pdf_combined, procs)
print_acceptance_totals(totals, procs)

                    
