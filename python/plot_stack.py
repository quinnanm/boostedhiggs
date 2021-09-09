import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

import argparse
import hist as hist2
import _pickle as cPickle

"""
For normal stack plots:
    python plot_stack.py --hname met_kin --haxis mt_lepmet --year 2017 --channel hadel --idir /uscms/home/docampoh/nobackup/boostedhiggs/hists/ --odir plots_Sep2
For cutflow stack plots:
    python plot_stack.py --hname cutflow --haxis cut --year 2017 --channel hadel --idir /uscms/home/docampoh/nobackup/boostedhiggs/hists/ --odir plots_Sep2
"""

def main(args):

    input_dir = args.idir + '/' + args.hist_name

    map_proc = {
        "hww": "GluGluHToWWToLNuQQ",
        "qcd": "QCD",
        "tt": "TTTo2L2Nu",
        "wjets": "WJetsToLNu",
    }
    
    paths_by_proc = {}
    
    if args.channel=="hadmu":
        paths_by_proc['data'] = "%s/muon_%s.pkl"%(input_dir,args.hist_name)
        data_key = "SingleMuon"
    elif args.channel=="hadel":
        paths_by_proc['data'] = "%s/electron_%s.pkl"%(input_dir,args.hist_name)
        data_key = "SingleElectron"
    else:
        print('Not a valid data channel')
        exit
    map_proc['data'] = data_key

    paths_by_proc['hww'] = "%s/hww_%s.pkl"%(input_dir,args.hist_name)

    mc_procs = ["tt","qcd","wjets"]
    for mc in mc_procs:
        paths_by_proc[mc] = "%s/%s_%s.pkl"%(input_dir,mc,args.hist_name)

    hists_by_proc = {}
    for key,path in paths_by_proc.items():
        with open(path, "rb") as f:
            pklf = cPickle.load(f)
        hists_by_proc[key] = pklf[map_proc[key]][args.hist_name][{"region":args.channel}].project(args.hist_axis)
        print(key,hists_by_proc[key])

    if args.hist_name == "cutflow":
        from utils import plot_cutflow
        plot_cutflow(
            data=hists_by_proc["data"],
            sig=hists_by_proc["hww"],
            bkg=[hists_by_proc["wjets"],
                 hists_by_proc["tt"],
                 hists_by_proc["qcd"]],
            bkg_labels=[r"$W(l\nu)$+jets", 
                        r"$t\bar{t}$ semileptonic", 
                        "QCD"],
            region=args.channel,
            odir=args.odir,
            year=args.year,
        )
    else:
        from utils import plot_stack
        plot_stack(
            data=hists_by_proc["data"],
            sig=hists_by_proc["hww"],
            bkg=[hists_by_proc["wjets"],
                 hists_by_proc["tt"],
                 hists_by_proc["qcd"]],
            bkg_labels=[r"$W(l\nu)$+jets", 
                        r"$t\bar{t}$ semileptonic", 
                        "QCD"],
            axis_name = args.hist_axis,
            region=args.channel,
            odir=args.odir,
            year=args.year,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hname',      dest='hist_name',  default=None,         help="hist name", type=str, required=True)
    parser.add_argument('--haxis',      dest='hist_axis',  default=None,         help="hist axis to project", type=str, required=True)
    parser.add_argument('--year',       dest='year',       default=None,         help="year", type=str, required=True)
    parser.add_argument('--channel',    dest='channel',    default=None,         help="channel (hadmu or hadel)", required=True)
    parser.add_argument('--idir',       dest='idir',       default=None,         help="directory with pickled histograms", required=True)
    parser.add_argument('--odir',       dest='odir',       default='plots/',     help="output directory with plots")

    args = parser.parse_args()

    import os
    os.system('mkdir -p %s'%args.odir)

    main(args)
