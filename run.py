#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import nanoevents
from coffea import processor
import time

import argparse
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
import pandas as pd
import os


def main(args):

    # make directory for output
    if not os.path.exists('./outfiles'):
        os.makedirs('./outfiles')

    channels = ["ele", "mu"]
    starti = args.starti
    job_name = '/' + str(starti)
    if args.n != -1:
        job_name += '-' + str(args.starti + args.n)

    # if --local is specefied in args, process only the args.sample provided
    if args.local:
        files = {}
        with open(f"fileset/pfnanoindex_{args.year}.json", 'r') as f:
            files_all = json.load(f)
            for subdir in files_all[args.year]:
                for key, flist in files_all[args.year][subdir].items():
                    if key in args.sample:
                        files[key] = ["root://cmsxrootd.fnal.gov/" + f for f in flist]
    else:
        # get samples
        if "metadata" in args.json:
            with open(args.json, 'r') as f:
                files = json.load(f)
        else:
            # hopefully this step is avoided in condor jobs that have metadata.json
            from condor.file_utils import loadJson
            files, _ = loadJson(args.json, args.year, args.pfnano)

    if not files:
        print('Did not find files.. Exiting.')
        exit(1)

    # build fileset with files to run per job
    fileset = {}
    for sample, flist in files.items():
        if args.sample:
            if sample not in args.sample.split(','):
                continue
        if args.n != -1:
            fileset[sample] = flist[args.starti:args.starti + args.n]
        else:
            fileset[sample] = flist

    print(len(list(fileset.keys())), 'Samples in fileset to be processed: ', list(fileset.keys()))
    print(fileset)

    # define processor
    if args.processor == 'hww':
        from boostedhiggs.hwwprocessor import HwwProcessor
        from boostedhiggs.trigger_efficiencies_processor import TriggerEfficienciesProcessor

        if 'APV' in args.year:
            p = HwwProcessor(year='2016', yearmod='APV', channels=channels,
                             inference=args.inference, output_location='./outfiles' + job_name)
        else:
            p = HwwProcessor(year=args.year, channels=channels, inference=args.inference,
                             output_location='./outfiles' + job_name)
    else:
        p = TriggerEfficienciesProcessor(year=args.year)

    tic = time.time()
    if args.executor == "dask":
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files="boostedhiggs",
        )
        client = Client(cluster)
        nanoevents_plugin = NanoeventsSchemaPlugin()
        client.register_worker_plugin(nanoevents_plugin)
        cluster.adapt(minimum=1, maximum=30)

        print("Waiting for at least one worker")
        client.wait_for_workers(1)

        # does treereduction help?
        executor = processor.DaskExecutor(status=True, client=client, treereduction=2)

    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        if args.executor == "futures":
            executor = processor.FuturesExecutor(status=True)
        else:
            executor = processor.IterativeExecutor(status=True)

    nanoevents.PFNanoAODSchema.mixins["SV"] = "PFCand"
    run = processor.Runner(
        executor=executor, savemetrics=True, schema=nanoevents.PFNanoAODSchema, chunksize=args.chunksize
    )

    out, metrics = run(
        fileset, "Events", processor_instance=p
    )

    elapsed = time.time() - tic
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

    # dump to pickle
    filehandler = open('./outfiles/' + job_name + '.pkl', "wb")
    pkl.dump(out, filehandler)
    filehandler.close()

    # merge parquet
    if args.processor == 'hww':
        for ch in channels:
            data = pd.read_parquet('./outfiles/' + job_name + ch + '/parquet')
            data.to_parquet('./outfiles/' + job_name + '_' + ch + '.parquet')

            # remove old parquet files
            os.system('rm -rf ./outfiles/' + job_name + ch)


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --json samples_pfnano.json
    # run locally on lpc as: python run.py --year 2017 --processor trigger --pfnano --n 1 --starti 0 --json samples_pfnano.json

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',        dest='year',           default='2017',
                        help="year",                                type=str)
    parser.add_argument('--starti',      dest='starti',         default=0,
                        help="start index of files",                type=int)
    parser.add_argument('--n',           dest='n',              default=-1,
                        help="number of files to process",          type=int)
    parser.add_argument('--json',        dest='json',           default="metadata.json",
                        help='path to datafiles',                   type=str)
    parser.add_argument('--sample',      dest='sample',         default=None,
                        help='specify sample',                      type=str)
    parser.add_argument("--processor",   dest="processor",      default="hww",
                        help="HWW processor",                       type=str)
    parser.add_argument("--chunksize",   dest='chunksize',      default=10000,
                        help="chunk size in processor",             type=int)
    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument("--local",       dest='local', action='store_true')
    parser.add_argument("--inference",   dest='inference', action='store_true')
    parser.add_argument("--no-inference", dest='inference', action='store_false')
    parser.add_argument("--pfnano",      dest='pfnano', action='store_true')
    parser.add_argument("--no-pfnano",   dest='pfnano', action='store_false')
    parser.set_defaults(pfnano=True)
    parser.set_defaults(inference=True)
    args = parser.parse_args()

    main(args)
