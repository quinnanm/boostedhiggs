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
    with open(args.fileset, 'r') as f:
        files = json.load(f)[args.sample]
    fileset = {}
    fileset[args.sample] = ["root://cmsxrootd.fnal.gov/"+ f for f in files[args.starti:args.endi]]
        
    # define processor
    from boostedhiggs.hwwprocessor import HwwProcessor
    p = HwwProcessor(year=int(args.year))

    if args.executor == "dask":
        import time
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files="src/HHbbVV",
        )
        client = Client(cluster)
        nanoevents_plugin = NanoeventsSchemaPlugin()
        client.register_worker_plugin(nanoevents_plugin)
        cluster.adapt(minimum=1, maximum=30)

        print("Waiting for at least one worker")
        client.wait_for_workers(1)

        # does treereduction help?
        executor = processor.DaskExecutor(status=True, client=client, treereduction=2)
        run = processor.Runner(
            executor=executor, savemetrics=True, schema=nanoevents.NanoAODSchema, chunksize=100000
        )
        out, metrics = run(
            {key: fileset[key] for key in args.samples}, "Events", processor_instance=p
        )
        elapsed = time.time() - tic
        print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")
    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        executor = (
            processor.futures_executor
            if args.executor == "futures"
            else processor.iterative_executor
        )

        exe_args = {
            "savemetrics": True,
            "schema": nanoevents.NanoAODSchema,
        }

        out, metrics = processor.run_uproot_job(
            {key: fileset[key] for key in args.samples},
            treename="Events",
            processor_instance=p,
            executor=executor,
            executor_args=exe_args,
            chunksize=args.chunksize,
        )

    filehandler = open(f"outfiles/{args.starti}-{args.endi}.pkl", "wb")
    pickle.dump(out, filehandler)
    filehandler.close()


if __name__ == "__main__":
    # e.g. 
    # inside a condor job: python run.py --year 2017 --processor hww --condor --starti 0 --endi 1 --fileset fileset_2017_UL_NANO.json --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8
    # inside a dask job:  python run.py --year 2017 --processor hww --dask --fileset fileset_2017_UL_NANO.json --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument("--processor",  dest="processor",  default="hww",        help="HWW processor", type=str)
    parser.add_argument("--dask",       dest="dask",       action="store_true",  default=False, help="Run with dask")
    parser.add_argument("--fileset",    dest="fileset",    default=None,         help="Fileset", required=True)    
    parser.add_argument('--sample',     dest='sample',     default=None,         help='sample name', required=True)
    parser.add_argument("--chunksize", type=int, default=2750, help="chunk size in processor")
    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    args = parser.parse_args()

    main(args)

