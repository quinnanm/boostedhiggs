from utils import axis_dict, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
from utils import simplified_labels, get_cutflow, get_xsecweight, get_sample_to_use, get_cutflow_axis

import yaml
import pickle as pkl
import pyarrow.parquet as pq
import numpy as np
import json
import os, glob, sys
import argparse

import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})

cut_keys = [
    "trigger",
    "leptonKin",
    "fatjetKin",
    "ht",
    "oneLepton",
    "notaus",
    "leptonInJet",
    "pre-sel",
]

global axis_dict
axis_dict["cutflow"] = get_cutflow_axis(cut_keys)
print('Cutflow with key names: ',cut_keys)

def make_hists(ch, idir, odir, vars_to_plot, weights, presel, samples):
    """
    Makes 1D histograms to be plotted as stacked over the different samples
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        vars_to_plot: the set of variables to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
        presel: pre-selection string
        weights: weights to be applied to MC
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json)
    """
    
    # define histograms
    hists = {}
    sample_axis = hist2.axis.StrCategory([], name="samples", growth=True)
    plot_vars = vars_to_plot
    plot_vars.append("cutflow")
    for var in plot_vars:
        hists[var] = hist2.Hist(
            sample_axis,
            axis_dict[var],
        )

    # cutflow dictionary
    cut_values = {}

    # pt cuts for variables
    pt_iso = {"ele": 120, "mu": 55}

    # loop over the samples
    for yr in samples.keys():

        # data label and lumi
        data_label = data_by_ch[ch]
        if yr == "2018":
            data_label = data_by_ch_2018[ch]
        f = open("../fileset/luminosity.json")
        luminosity = json.load(f)[data_label][yr]
        f.close()
        print(f"Processing samples from year {yr} with luminosity {luminosity} for channel {ch}")

        for sample in samples[yr][ch]:
            print(f"Sample {sample}")

            # check if the sample was processed
            pkl_dir = f"{idir}_{yr}/{sample}/outfiles/*.pkl"
            pkl_files = glob.glob(pkl_dir)
            if not pkl_files:  # skip samples which were not processed
                print("- No processed files found...", pkl_dir, "skipping sample...", sample)
                continue

            # get list of parquet files that have been post processed
            parquet_files = glob.glob(f"{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet")

            # define an is_data boolean
            is_data = False
            if data_label in sample:
                is_data = True

            # get combined sample
            sample_to_use = get_sample_to_use(sample,yr)

            # get cutflow
            xsec_weight = get_xsecweight(pkl_files,yr,sample,is_data,luminosity)
            if sample_to_use not in cut_values.keys():
                cut_values[sample_to_use] = dict.fromkeys(cut_keys,0)
            cutflow = get_cutflow(cut_keys,pkl_files,yr,sample,xsec_weight,ch)
            for key,val in cutflow.items():
                cut_values[sample_to_use][key] += val

            cut_values[sample_to_use]["pre-sel"] = 0

            sample_yield = 0
            for parquet_file in parquet_files:
                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    if is_data:
                        print("Not able to read data: ", parquet_file, " should remove events from scaling/lumi")
                    else:
                        print("Not able to read data from ", parquet_file)
                    continue

                # print parquet content
                # print(data.columns)

                if len(data) == 0:
                    print(f"WARNING: Parquet file empty {yr} {ch} {sample} {parquet_file}")
                    continue

                # modify dataframe with pre-selection query
                if presel is not None:
                    data = data.query(presel)

                if not is_data:
                    event_weight = xsec_weight
                    weight_ones = np.ones_like(data["weight_genweight"])
                    for w in weights:
                        try:
                            event_weight *= data[w]
                        except:
                            if w!="weight_vjets_nominal":
                                print(f"No {w} variable in parquet for sample {sample}")
                            # break
                else:
                    weight_ones = np.ones_like(data["lep_pt"])
                    event_weight = weight_ones
                    
                # account yield for extra selection 
                # (with only the xsec weight)
                # sample_yield += np.sum(weight_ones*xsec_weight)
                sample_yield += np.sum(event_weight)

                for var in plot_vars:
                    if var == "cutflow":
                        continue
                    if var == "score" and not args.add_score:
                        continue

                    # for specific variables introduce cut
                    if "lowpt" in var:
                        select_var = (data["lep_pt"] < pt_iso[ch])
                    elif "highpt" in var:
                        select_var = (data["lep_pt"] > pt_iso[ch])
                    else:
                        select_var = (data["lep_pt"] > 0)
                        
                    var_plot = var.replace('_lowpt', '').replace('_highpt', '')
                    if var_plot not in data.keys():
                        if 'gen' in var: continue
                        print(f"Var {var} not in parquet keys")
                        continue

                    # filling histograms
                    hists[var].fill(
                        samples=sample_to_use,
                        var=data[var_plot][select_var],
                        weight=event_weight[select_var],
                    )

            cut_values[sample_to_use]["pre-sel"] += sample_yield

            # fill cutflow histogram once we have all the values
            for key, numevents in cut_values[sample_to_use].items():
                cut_index = cut_keys.index(key)
                hists["cutflow"].fill(
                    samples=sample_to_use,
                    var=cut_index,
                    weight=numevents
                )

            #samples = [hists["cutflow"].axes[0].value(i) for i in range(len(hists["cutflow"].axes[0].edges))]
            #print(sample,samples)

        # save cutflow values
        with open(f"{odir}/cut_values_{ch}.pkl", "wb") as f:
            pkl.dump(cut_values, f)

    samples = [hists["cutflow"].axes[0].value(i) for i in range(len(hists["cutflow"].axes[0].edges))]
    print(samples)

    # store the hists variable
    with open(f"{odir}/{ch}_hists.pkl", "wb") as f:
        pkl.dump(hists, f)

def main(args):
    # append '/year' to the output directory
    odir = args.odir + "/" + args.year
    if not os.path.exists(odir):
        os.system(f'mkdir -p {odir}')

    channels = args.channels.split(",")

    # get year
    years = ["2016", "2016APV", "2017", "2018"] if args.year == "Run2" else [args.year]

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    # load yaml config file
    with open(args.vars) as f:
        variables = yaml.safe_load(f)

    print(variables)

    # variables to plot
    vars_to_plot = {}
    # list of weights to apply to MC
    weights = {}
    # pre-selection string
    presel = {}
    for ch in variables.keys():
        vars_to_plot[ch] = []
        for key, value in variables[ch]["vars"].items():
            if value == 1:
                vars_to_plot[ch].append(key)

        weights[ch] = []
        for key, value in variables[ch]["weights"].items():
            if value == 1:
                weights[ch].append(key)

        presel_str = None
        if type(variables[ch]["pre-sel"]) is list:
            presel_str = variables[ch]["pre-sel"][0]
            for i,sel in enumerate(variables[ch]["pre-sel"]):
                if i==0: continue
                presel_str += f'& {sel}'
        presel[ch] = presel_str

    os.system(f"cp {args.vars} {odir}/")
     
    for ch in channels:
        if len(glob.glob(f"{odir}/{ch}_hists.pkl")) > 0:
            print("Histograms already exist - remaking them")
        print(f"Making histograms for {ch}...")
        print("Weights: ",weights[ch])
        print("Pre-selection: ",presel[ch])
        make_hists(ch, args.idir, odir, vars_to_plot[ch], weights[ch], presel[ch], samples)

if __name__ == "__main__":
    # e.g.
    # run locally as: python make_hists.py --year 2017 --odir Nov4 --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Nov4
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", dest="year", required=True, choices=["2016", "2016APV", "2017", "2018", "Run2"], help="year"
    )
    parser.add_argument(
        "--vars", dest="vars", default="plot_configs/vars.yaml", help="path to json with variables to be plotted"
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        default="plot_configs/samples_pfnano.json",
        help="path to json with samples to be plotted",
    )
    parser.add_argument("--channels", dest="channels", default="ele,mu", help="channels for which to plot this variable")
    parser.add_argument(
        "--odir", dest="odir", default="hists", help="tag for output directory... will append '_{year}' to it"
    )
    parser.add_argument("--idir", dest="idir", default="../results/", help="input directory with results - without _{year}")
    parser.add_argument("--add_score", dest="add_score",  action="store_true", help="Add inference score")

    args = parser.parse_args()

    main(args)
