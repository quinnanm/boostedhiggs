#!/usr/bin/python

import os, sys
import json
import uproot
import awkward as ak
import numpy as np

import argparse

# How you would run this
# e.g. python run.py --year 2017 --starti 0 --endi 1 --samples BulkGravTohhTohVVhbb_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia8

def main(year,samples,starti,endi):

    from coffea import processor, util, hist
    from boostedhiggs.bbwwprocessor import HHbbWW

    p = HHbbWW(year=year)
    from coffea.nanoevents import NanoAODSchema
    nargs = {
        "schema": NanoAODSchema,
    }
    NanoAODSchema.mixins["FatJetLS"] = "PtEtaPhiMLorentzVector"

    files = {}
    with open('data/fileset_2017_das.json', 'r') as f:
        newfiles = json.load(f)
        files.update(newfiles)

    selfiles = {k: files[k][starti:endi] for k in samples}
    out = processor.run_uproot_job(selfiles,"Events",p,processor.futures_executor,nargs,chunksize=50000)

    util.save(out, 'hhbbww.coffea')

    return

if __name__ == "__main__":
    #ex. python run.py --year 2018 --starti 0 --endi -1 --samples HHBBWW
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year",        type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index",   type=int)
    parser.add_argument('--samples',    dest='samples',    default=[],           help='samples',     nargs='+')
    args = parser.parse_args()

    main(args.year,args.samples,args.starti,args.endi)
