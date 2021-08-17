#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor
import pickle

import argparse
import warnings


def main(args):

    # read samples to submit
    # TODO: get this to a json that can be identified by year and sample
    with open(args.fileset, 'r') as f:
        files = json.load(f)[args.sample]
    fileset = files[args.starti:args.endi]

    # define processor
    if args.processor == "hww":
        from boostedhiggs import HwwProcessor
        # TODO: add arguments to processor
        p = HwwProcessor()
    else:
        warnings.warn('Warning: no processor declared')
        return

    if args.condor:
        uproot.open.defaults['xrootd_handler'] = uproot.source.xrootd.MultithreadedXRootDSource

        exe_args = {'savemetrics':True,
                    'schema':NanoAODSchema,
                    'retries': 1}

        out, metrics = processor.run_uproot_job(
            fileset,
            'Events',
            p,
            processor.futures_executor,
            exe_args,
            chunksize=10000,
        )

        print(f"num: {out['num'].view(flow=True)}")
        print(f"den: {out['den'].view(flow=True)}")
        print(f"Metrics: {metrics}")

        filehandler = open(f'outfiles/{args.year}_{args.sample}_{args.starti}-{args.endi}.hist', 'wb')
        pickle.dump(out, filehandler)
        filehandler.close()
        
if __name__ == "__main__":
    # e.g. 
    # inside a condor job: python run.py --year 2017 --processor hww --condor --starti 0 --endi 1 --fileset fileset_2017_UL_NANO.json --sample GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument("--processor",  dest="processor",  default="hww",        help="HWW processor", type=str)
    parser.add_argument("--condor",     dest="condor",     action="store_true",  default=True,  help="Run with condor")
    parser.add_argument("--fileset",    dest="fileset",    default=None,         help="Fileset", required=True)
    parser.add_argument('--sample',     dest='sample',     default=None,         help='sample name', required=True)
    args = parser.parse_args()

    main(args)
