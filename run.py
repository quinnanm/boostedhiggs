#!/usr/bin/python

import os, sys
import json
import uproot
import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster
import awkward as ak
import numpy as np

import argparse

# How you would run this
# e.g. python run.py --year 2017 --starti 0 --endi 1 --samples BulkGravTohhTohVVhbb_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia8

def main(year,samples,starti,endi):

    from coffea import processor, util, hist, nanoevents
    from boostedhiggs.bbwwprocessor import HHbbWW

    tic = time.time()
    cluster = LPCCondorCluster(
        ship_env=True,
        transfer_input_files="boostedhiggs",
    )
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=50)
    client = Client(cluster)

    nanoevents.NanoAODSchema.mixins["FatJetLS"] = "PtEtaPhiMLorentzVector"

    def patch_fatjets():
        from coffea import nanoevents
        nanoevents.NanoAODSchema.mixins["FatJetLS"] = "PtEtaPhiMLorentzVector"

    client.register_worker_callbacks(patch_fatjets)

    exe_args = {
        "client": client,
        "savemetrics": True,
        "schema": nanoevents.NanoAODSchema,
        "align_clusters": True,
    }

    p = HHbbWW(year=year)

    print("Waiting for at least one worker...")
    client.wait_for_workers(1)
    out, metrics = processor.run_uproot_job(
        "data/fileset_2017UL.json",
        treename="Events",
        processor_instance=p,
        executor=processor.dask_executor,
        executor_args=exe_args,
    )

    elapsed = time.time() - tic
    print(f"Output: {out}")
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")
    print(f"Events/s: {metrics['entries'] / elapsed:.0f}")

    util.save(out, 'hhbbww.coffea')

    return

if __name__ == "__main__":
    #ex. python run.py --year 2018 --starti 0 --endi -1 --samples HHBBWW
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year",        type=str)
    args = parser.parse_args()

    main(args.year,[],0,-1)
