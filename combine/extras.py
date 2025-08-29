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
genxsecs = {'vbf': 0.8082134, 'ggf1': 0.10078092000000001, 'ggf2': 0.10078092000000001, 'ggf3': 0.10078092000000001}
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



def inspect_ps_convention(proc, year, lep='mu', sample_rows=10):
    base_col = f'weight_{lep}_genweight'
    ps_cols = {
        'PSISRUp':  f'weight_{lep}_PSISRUp',
        'PSISRDown':f'weight_{lep}_PSISRDown',
        'PSFSRUp':  f'weight_{lep}_PSFSRUp',
        'PSFSRDown':f'weight_{lep}_PSFSRDown',
    }
    cols = [base_col] + list(ps_cols.values())

    files = getfilelist(proc, year, f'{lep}.parquet')
    if not files:
        print(f"No files for proc={proc}, year={year}")
        return

    # load a single file (fast check)
    df = pd.read_parquet(files[0], columns=cols)
    # keep only rows where base != 0 to avoid ratio explosions
    df = df[df[base_col].abs() > 1e-12].copy()

    # sample a handful to print
    view = df.sample(n=min(sample_rows, len(df)), random_state=123).copy()

    # compute ratios (ps / base) and products (base * ps) for inspection
    for tag, col in ps_cols.items():
        view[f'{tag}_ratio_to_base'] = view[col] / view[base_col]
        view[f'{tag}_times_base']   = view[col] * view[base_col]

    # pretty print a few columns
    show_cols = [base_col] + [c for tag in ps_cols for c in (ps_cols[tag], f'{tag}_ratio_to_base')]
    print("Sample rows (base, PS, PS/base):")
    print(view[show_cols].head(sample_rows).to_string(index=False))

    # heuristics to guess convention
    def pct_in_band(series, lo=0.5, hi=2.0):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return 0.0
        sabs = s.abs()
        return ( (sabs>=lo) & (sabs<=hi) ).mean()

    print("\nHeuristic summary (fraction in [0.5, 2.0]):")
    guesses = {}
    for tag, col in ps_cols.items():
        frac_ps   = pct_in_band(df[col])                          # are PS values ~ O(1)?
        frac_ratio= pct_in_band(df[col] / df[base_col])           # is PS/base ~ O(1)?
        print(f"{tag:9s}  PS~O(1): {frac_ps:5.2%}   (PS/base)~O(1): {frac_ratio:5.2%}")

        # If PS itself ~ O(1) and (PS/base) is NOT, likely multiplicative factor.
        # If (PS/base) ~ O(1) and PS is NOT, likely absolute already includes base.
        if   frac_ps > 0.7 and frac_ratio < 0.3:
            guess = "multiplicative factor (use base * PS)"
        elif frac_ratio > 0.7 and frac_ps < 0.3:
            guess = "absolute weight (already includes base)"
        elif frac_ps > frac_ratio:
            guess = "likely factor"
        elif frac_ratio > frac_ps:
            guess = "likely absolute"
        else:
            guess = "inconclusive"
        guesses[tag] = guess

    print("\nConvention guess per branch:")
    for tag, g in guesses.items():
        print(f"  {tag}: {g}")

    print("\nInterpretation:")
    print("- If PS values themselves are ~1 (and PS/base is all over), treat them as multiplicative factors and use base_weight * PS for sums.")
    print("- If PS/base ~1 (and PS values mirror the scale/sign of base), treat PS as absolute already multiplied weights and use PS directly for sums.")


def inspect_scale_convention(proc, year, lep='mu', sample_rows=10, rtol=0.05):
    """
    Quick validation of whether weight_scale4 is an ABSOLUTE weight (≈ base gen weight)
    or a MULTIPLICATIVE factor (~1).

    Heuristics reported:
      - frac(|scale4/base - 1| < rtol)  -> high => ABSOLUTE
      - frac(|scale4 - 1| < rtol)       -> high => FACTOR
    Also shows a few sample rows and compares other scale weights to scale4.
    """
    base_col  = f'weight_{lep}_genweight'
    scale_nom = 'weight_scale4'
    scale_vars = ['weight_scale0','weight_scale1','weight_scale3',
                  'weight_scale5','weight_scale7','weight_scale8']
    cols = [base_col, scale_nom] + scale_vars

    files = getfilelist(proc, year, f'{lep}.parquet')
    if not files:
        print(f"[inspect_scale_convention] No files for {proc} {year}")
        return

    # Read one file (fast) and keep finite, nonzero base
    df = pd.read_parquet(files[0], columns=[c for c in cols if c in pd.read_parquet(files[0]).columns])
    for c in cols:
        if c not in df:
            print(f"[WARN] Column {c} missing, creating placeholder 1.0")
            df[c] = 1.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[base_col, scale_nom]).copy()
    df = df[df[base_col].abs() > 1e-12]
    if len(df) == 0:
        print("[inspect_scale_convention] No usable rows after filtering.")
        return

    # Ratios
    ratio_scale4_to_base = df[scale_nom] / df[base_col]
    # “Close to X” helpers
    def frac_close_to(series, target, rtol):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            return 0.0
        rel = (s - target) / (np.where(target != 0, target, 1.0))
        return (rel.abs() < rtol).mean()

    frac_scale4_eq_base = frac_close_to(ratio_scale4_to_base, 1.0, rtol)
    frac_scale4_eq_one  = frac_close_to(df[scale_nom], 1.0, rtol)

    # Summary for other scale weights vs nominal (if absolute, these are factors)
    relatives = {}
    for c in scale_vars:
        if c in df:
            rel = (df[c] / df[scale_nom]).replace([np.inf,-np.inf], np.nan).dropna()
            if len(rel):
                relatives[c] = {
                    'median': float(np.median(rel)),
                    'p16':    float(np.percentile(rel, 16)),
                    'p84':    float(np.percentile(rel, 84)),
                }

    # Print a small sample for eyeballing
    view_cols = [base_col, scale_nom] + scale_vars
    print("\nSample rows (base vs scale4 and a few scale vars):")
    print(df[view_cols].head(sample_rows).to_string(index=False))

    # Decision
    print("\nHeuristic checks (rtol = {:.1f}%):".format(100*rtol))
    print(f"  frac(|scale4/base - 1| < rtol)  = {frac_scale4_eq_base:6.2%}  (ABSOLUTE if high)")
    print(f"  frac(|scale4 - 1| < rtol)       = {frac_scale4_eq_one:6.2%}  (FACTOR if high)")

    if frac_scale4_eq_base > 0.80 and frac_scale4_eq_one < 0.30:
        guess = "ABSOLUTE: use weight_scale4 as nominal weight"
    elif frac_scale4_eq_one > 0.80 and frac_scale4_eq_base < 0.30:
        guess = "FACTOR: use base * weight_scale4 as nominal"
    else:
        guess = "INCONCLUSIVE: mixed; inspect sample rows/statistics"

    print(f"\nConvention guess: {guess}")

    if relatives:
        print("\nOther scale weights relative to scale4 (median [p16, p84]):")
        for c, stats in relatives.items():
            print(f"  {c:14s}: {stats['median']:.3f}  [{stats['p16']:.3f}, {stats['p84']:.3f}]")

    print("\nRules of thumb:")
    print("  • If ABSOLUTE:  A_nom = sum(scale4_reco)/sum(scale4_gen); variations use sum(scaleX_*) directly.")
    print("  • If FACTOR:    A_nom = sum(base_reco)/sum(base_gen);   variations use sum(base_* * scaleX_*).")



def inspect_pdf_alphas_convention(proc, year, lep='mu', sample_rows=8, rtol=0.05):
    """
    Decide if weight_pdf{i} (0..100) and weight_pdf101/102 are
    multiplicative factors (~1) or absolute weights (~base*factor).

    Heuristics:
      - If mean frac(|pdf - 1| < rtol) is high and mean frac(|pdf/base - 1| < rtol) is low
        -> FACTOR (multiply base by pdf).
      - If mean frac(|pdf/base - 1| < rtol) is high and mean frac(|pdf - 1| < rtol) is low
        -> ABSOLUTE (use pdf directly).
    """
    base_col = f'weight_{lep}_genweight'
    pdf_cols  = [f'weight_pdf{i}' for i in range(0, 101)]   # replicas
    alphas_cols = [f'weight_pdf{i}' for i in (101, 102)]     # αs up/down
    all_cols = [base_col] + pdf_cols + alphas_cols

    files = getfilelist(proc, year, f'{lep}.parquet')
    if not files:
        print(f"[inspect_pdf_alphas_convention] No files for {proc} {year}")
        return

    # Read one file and keep available columns
    df0 = pd.read_parquet(files[0])
    keep = [c for c in all_cols if c in df0.columns]
    missing = [c for c in all_cols if c not in df0.columns]
    if missing:
        print("[WARN] missing columns:", missing)
    df = df0[keep].copy()

    # Filter to finite, nonzero base
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[base_col])
    df = df[df[base_col].abs() > 1e-12]
    if len(df) == 0:
        print("No usable rows after filtering.")
        return

    # helper
    def frac_close_to(series, target, rtol):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0: return 0.0
        return ( (s - target).abs() <= rtol*max(1.0, abs(target)) ).mean()

    # compute stats per column
    stats = {}
    have_pdf = []
    have_alphas = []
    for c in pdf_cols + alphas_cols:
        if c not in df: continue
        r_pdf  = frac_close_to(df[c], 1.0, rtol)                 # is pdf ≈ 1?
        r_base = frac_close_to(df[c] / df[base_col], 1.0, rtol)  # is pdf/base ≈ 1?
        stats[c] = (r_pdf, r_base)
        (have_pdf if c in pdf_cols else have_alphas).append(c)

    # summarize
    def summarize(keys):
        if not keys: return (0.0, 0.0)
        arr_pdf  = np.array([stats[k][0] for k in keys])
        arr_base = np.array([stats[k][1] for k in keys])
        return float(arr_pdf.mean()), float(arr_base.mean())

    pdf_pdf_mean,  pdf_base_mean  = summarize(have_pdf)
    aS_pdf_mean,   aS_base_mean   = summarize(have_alphas)

    # sample view
    view_cols = [base_col] + [c for c in (['weight_pdf0','weight_pdf1','weight_pdf50','weight_pdf100',
                                           'weight_pdf101','weight_pdf102']) if c in df.columns]
    print("\nSample rows:")
    print(df[view_cols].head(sample_rows).to_string(index=False))

    print("\nHeuristic checks (rtol = {:.1f}%):".format(100*rtol))
    print(f"PDF replicas 0–100:  mean frac(|pdf-1|<rtol)={pdf_pdf_mean:5.2%}   "
          f"mean frac(|pdf/base-1|<rtol)={pdf_base_mean:5.2%}")
    print(f"alpha_s 101–102:    mean frac(|pdf-1|<rtol)={aS_pdf_mean:5.2%}    "
          f"mean frac(|pdf/base-1|<rtol)={aS_base_mean:5.2%}")

    def guess(name, m_pdf, m_base):
        if m_pdf > 0.80 and m_base < 0.30:  return f"{name}: FACTOR (multiply base * weight_pdf*)"
        if m_base > 0.80 and m_pdf < 0.30:  return f"{name}: ABSOLUTE (use weight_pdf* directly)"
        return f"{name}: INCONCLUSIVE (mixed; eyeball sample rows)"

    print("\nConvention guess:")
    print("  " + guess("PDF", pdf_pdf_mean, pdf_base_mean))
    print("  " + guess("alpha_s", aS_pdf_mean, aS_base_mean))

    print("\nRules of thumb:")
    print("  • If FACTOR:  A_nom = sum(base)_reco / sum(base)_gen; "
          "A_var(k) = sum(base*pdf_k)_reco / sum(base*pdf_k)_gen; "
          "combine replicas via PDF4LHC (RMS or Hessian), αs from 101/102 directly.")
    print("  • If ABSOLUTE: A_nom = sum(pdf_nom?)_reco / sum(pdf_nom?)_gen (usually still base as nominal); "
          "A_var(k) = sum(pdf_k)_reco / sum(pdf_k)_gen.")

###### execute
#inspect_ps_convention(proc='ggf1', year='2018', lep='mu', sample_rows=10)
#inspect_scale_convention('ggf1', '2018', lep='mu', sample_rows=8, rtol=0.05)
inspect_pdf_alphas_convention('ggf1', '2018', lep='mu', sample_rows=8, rtol=0.05)