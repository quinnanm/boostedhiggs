#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018
from utils import get_simplified_label, get_sum_sumgenweight
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_stacked_hists_years(tag, odir, vars_to_plot, samples, years, channels, pfnano):
    """
    Makes 1D histograms to be plotted as stacked over the different samples
    Args:
        vars_to_plot: the set of variables to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
    """

    # initialize the histograms for the different channels and different variables
    hists = {}
    for ch in channels:
        hists[ch] = {}
        for var in vars_to_plot[ch]:
            year_axis = hist2.axis.StrCategory([], name='years', growth=True)
            sample_axis = hist2.axis.StrCategory([], name='samples', growth=True)
            cut_axis = hist2.axis.StrCategory([], name='cuts', growth=True)

            hists[ch][var] = hist2.Hist(
                year_axis,
                sample_axis,
                cut_axis,
                axis_dict[var],
            )

    for year in years:
        idir = tag + '_' + year

        # Get luminosity per year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        data_label = data_by_ch

        xsec_weight_by_sample = {}
        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                is_data = False

                for key in data_label.values():
                    if (key in sample) or ('EGamma' in sample):
                        is_data = True

                if not is_data and sample not in xsec_weight_by_sample.keys():
                    pkl_dir = f'{idir}/{sample}/outfiles/*.pkl'
                    pkl_files = glob.glob(pkl_dir)  # get list of files that were processed
                    if not pkl_files:  # skip samples which were not processed
                        print('- No processed files found...', pkl_dir, 'skipping sample...', sample)
                        continue

                    # Find xsection
                    f = open('../fileset/xsec_pfnano.json')

                    xsec = json.load(f)
                    f.close()
                    xsec = eval(str((xsec[sample])))

                    # Get sum_sumgenweight of sample
                    sum_sumgenweight = get_sum_sumgenweight(idir, year, sample)

                    # Get overall weighting of events
                    # each event has (possibly a different) genweight... sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
                    xsec_weight = (xsec * luminosity[year]) / (sum_sumgenweight)
                    xsec_weight_by_sample[sample] = xsec_weight

                elif sample in xsec_weight_by_sample.keys():
                    xsec_weight = xsec_weight_by_sample[sample]

                else:
                    xsec_weight = 1

                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed

                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample', sample)

                for parquet_file in parquet_files:
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue

                    for var in vars_to_plot[ch]:
                        if var not in data.keys():
                            # print(f'- No {var} for {year}/{ch} - skipping')
                            continue
                        if len(data) == 0:
                            continue

                        # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                        data = data[data[var] != -1]

                        try:
                            event_weight = data['weight']
                        except:
                            data['weight'] = 1  # for data fill a weight column with ones

                        # filling histograms
                        single_sample = None
                        for single_key, key in add_samples.items():
                            if key in sample:
                                single_sample = single_key

                        # combining all pt bins of a specefic process under one name
                        if single_sample is not None:
                            hists[ch][var].fill(
                                years=year,
                                samples=single_sample,
                                cuts='preselection',
                                var=data[var],
                                weight=xsec_weight * data['weight'],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=single_sample,
                                cuts='btag',
                                var=data[var][data["anti_bjettag"] == 1],
                                weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=single_sample,
                                cuts='dr',
                                var=data[var][data["leptonInJet"] == 1],
                                weight=xsec_weight * data['weight'][data["leptonInJet"] == 1],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=single_sample,
                                cuts='btagdr',
                                var=data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                                weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            )
                        # otherwise give unique name
                        else:
                            hists[ch][var].fill(
                                years=year,
                                samples=sample,
                                cuts='preselection',
                                var=data[var],
                                weight=xsec_weight * data['weight'],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=sample,
                                cuts='btag',
                                var=data[var][data["anti_bjettag"] == 1],
                                weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=sample,
                                cuts='dr',
                                var=data[var][data["leptonInJet"] == 1],
                                weight=xsec_weight * data['weight'][data["leptonInJet"] == 1],
                            )
                            hists[ch][var].fill(
                                years=year,
                                samples=sample,
                                cuts='btagdr',
                                var=data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                                weight=xsec_weight * data['weight'][data["anti_bjettag"] == 1][data["leptonInJet"] == 1],
                            )

    # TODO: combine histograms for all years here and flag them as year='combined'

    # store the hists variable
    with open(f'{odir}/stacked_hists.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_stacked_hists_years(odir, vars_to_plot, years, channels, pfnano, cut='preselection', logy=True, add_data=True):
    """
    Plots the stacked 1D histograms that were made by "make_stacked_hists" function
    Args:
        vars_to_plot: the set of variable to plot a 1D-histogram of (by default: the samples with key==1 defined in plot_configs/vars.json)
        samples: the set of samples to run over (by default: the samples with key==1 defined in plot_configs/samples_pfnano.json)
        cut: the cut to apply when plotting the histogram... choices are ['preselection', 'dr', 'btag', 'btagdr']
    """

    print(f'plotting for {cut} cut')

    # load the hists
    with open(f'{odir}/stacked_hists.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    if logy:
        if not os.path.exists(f'{odir}/hists_log'):
            os.makedirs(f'{odir}/hists_log')
    else:
        if not os.path.exists(f'{odir}/hists'):
            os.makedirs(f'{odir}/hists')

    # make the histogram plots in this directory
    for ch in channels:
        data_label = data_by_ch[ch]

        for var in vars_to_plot[ch]:
            if hists[ch][var].shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
                continue

            # get histograms
            h = hists[ch][var]

            # get samples existing in histogram
            samples = [h.axes[1].value(i) for i in range(len(h.axes[1].edges))]
            signal_labels = [label for label in samples if label in signal_by_ch[ch]]
            bkg_labels = [label for label in samples if (label and label != data_label and label not in signal_labels)]

            # data
            data = None
            if data_label in samples:
                data = h[{"samples": data_label, 'cuts': cut}]

            # signal
            signal = [h[{"samples": label, "cuts": cut}[{'years': years}][{'years': sum}]] for label in signal_labels]
            if not logy:
                signal = [s * 10 for s in signal]  # if not log, scale the signal

            # background
            bkg = [h[{"samples": label, "cuts": cut}] for label in bkg_labels]

            # print(data,signal,bkg)

            if add_data and data and len(bkg) > 0:
                fig, (ax, rax) = plt.subplots(nrows=2,
                                              ncols=1,
                                              figsize=(8, 8),
                                              tight_layout=True,
                                              gridspec_kw={"height_ratios": (3, 1)},
                                              sharex=True
                                              )
                fig.subplots_adjust(hspace=.07)
                rax.errorbar(
                    x=[data.axes.value(i)[0] for i in range(len(data.values()))],
                    y=data.values() / np.sum([b.values() for b in bkg], axis=0),
                    fmt="ko",
                )
                # NOTE: change limit later
                rax.set_ylim(0., 1.2)
            else:
                fig, ax = plt.subplots(1, 1)

            if len(bkg) > 0:
                hep.histplot(bkg,
                             ax=ax,
                             stack=True,
                             sort='yield',
                             histtype="fill",
                             label=[get_simplified_label(bkg_label) for bkg_label in bkg_labels],
                             )
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    handle.set_color(color_by_sample[label])
            if add_data and data:
                data_err_opts = {
                    'linestyle': 'none',
                    'marker': '.',
                    'markersize': 12.,
                    'elinewidth': 2,
                }
                hep.histplot(data,
                             ax=ax,
                             histtype="errorbar",
                             color="k",
                             yerr=True,
                             label=get_simplified_label(data_label),
                             **data_err_opts
                             )

            if len(signal) > 0:
                hep.histplot(signal,
                             ax=ax,
                             label=[get_simplified_label(sig_label) for sig_label in signal_labels],
                             color='red'
                             )

            if logy:
                ax.set_yscale('log')
                ax.set_ylim(0.1)
            ax.set_title(f'{ch} channel \n with {cut} cut')
            ax.legend()

            hep.cms.lumitext(f"combined (13 TeV)", ax=ax)
            hep.cms.text("Work in Progress", ax=ax)

            if logy:
                print('Saving to ', f'{odir}/hists_log/{var}_{ch}_{cut}.pdf')
                plt.savefig(f'{odir}/hists_log/{var}_{ch}_{cut}.pdf')
            else:
                print('Saving to ', f'{odir}/hists/{var}_{ch}_{cut}.pdf')
                plt.savefig(f'{odir}/hists/{var}_{ch}_{cut}.pdf')
            plt.close()


def main(args):
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    years = args.years.split(',')
    channels = args.channels.split(',')

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

    # get variables to plot
    f = open(args.vars)
    variables = json.load(f)
    f.close()
    vars_to_plot = {}
    for ch in variables.keys():
        vars_to_plot[ch] = []
        for key, value in variables[ch].items():
            if value == 1:
                vars_to_plot[ch].append(key)

    if args.make_hists:
        print('Making histograms...')
        make_stacked_hists_years(args.idir, args.odir, vars_to_plot, samples, years, channels, args.pfnano)

    if args.plot_hists:
        print('Plotting histograms...')

        for ch in channels:
            if ch == 'had':
                cuts = ['preselection']
            else:
                cuts = ['preselection', 'btagdr']

            for cut in cuts:
                plot_stacked_hists_years(args.odir, vars_to_plot, years, [ch], args.pfnano, cut, logy=True)
                # plot_stacked_hists_years(args.odir, vars_to_plot, years, ch, args.pfnano, cut, logy=False)


if __name__ == "__main__":
    # e.g.
    # run locally as:  python merge_years_hists.py --years 2016,2017,2018 --odir hists/stacked_hists --pfnano --make_hists --plot_hists --channels ele --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',            dest='years',          default='2017',                               help="year")
    parser.add_argument('--vars',             dest='vars',           default="plot_configs/vars.json",             help='path to json with variables to be plotted')
    parser.add_argument('--samples',          dest='samples',        default="plot_configs/samples_pfnano.json",   help='path to json with samples to be plotted')
    parser.add_argument('--channels',         dest='channels',       default='ele,mu,had',                         help='channels for which to plot this variable')
    parser.add_argument('--odir',             dest='odir',           default='hists',                              help="tag for output directory")
    parser.add_argument('--idir',             dest='idir',           default='../results/',                        help="input directory with results")
    parser.add_argument("--pfnano",           dest='pfnano',         action='store_true',                          help="Run with pfnano")
    parser.add_argument("--make_hists",       dest='make_hists',     action='store_true',                          help="Make hists")
    parser.add_argument("--plot_hists",       dest='plot_hists',     action='store_true',                          help="Plot the hists")

    args = parser.parse_args()

    main(args)
