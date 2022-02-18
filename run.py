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
from BoolArg import BoolArg


def main(args):

    # make directory for output
    if not os.path.exists('./outfiles'):
        os.makedirs('./outfiles')

    channels = ["ele", "mu", "had"]
    job_name = '/' + str(args.starti) + '-' + str(args.endi)

    # get samples
    if args.pfnano:
        fname = f"data/pfnanoindex_{args.year}.json"

        f = open("samples_config_pfnano.json")
        json_samples = json.load(f)
        f.close()
    else:
        fname = f"data/fileset_{args.year}_UL_NANO.json"

        f = open("samples_config.json")
        json_samples = json.load(f)
        f.close()

    samples = []
    for key, value in json_samples.items():
        if value == 1:
            samples.append(key)
        if not args.all:
            break

    fileset = {}

    with open(fname, 'r') as f:
        if args.pfnano:
            files = json.load(f)[args.year]
            for s in samples:
                for subdir in files:
                    for key, flist in files[subdir].items():
                        if s in key:
                            print('s', s)
                            fileset[s] = ["root://cmsxrootd.fnal.gov/" + f for f in files[subdir][key][args.starti:args.endi]]
        else:
            files = json.load(f)
            for s in samples:
                fileset[s] = ["root://cmsxrootd.fnal.gov/" + f for f in files[s][args.starti:args.endi]]

    # define processor
    if args.processor == 'hww':
        from boostedhiggs.hwwprocessor import HwwProcessor
        p = HwwProcessor(year=args.year, channels=channels, output_location='./outfiles' + job_name)
    else:
        from boostedhiggs.trigger_efficiencies_processor import TriggerEfficienciesProcessor
        p = TriggerEfficienciesProcessor(year=int(args.year))

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
        run = processor.Runner(
            executor=executor, savemetrics=True, schema=nanoevents.NanoAODSchema, chunksize=100000
        )
    else:
        uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

        if args.executor == "futures":
            executor = processor.FuturesExecutor(status=True)
        else:
            executor = processor.IterativeExecutor(status=True)
        run = processor.Runner(
            executor=executor, savemetrics=True, schema=nanoevents.NanoAODSchema, chunksize=args.chunksize
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

    # # merge parquet
    for ch in channels:
        data = pd.read_parquet('./outfiles/' + job_name + ch + '/parquet')
        data.to_parquet('./outfiles/' + job_name + '_' + ch + '.parquet')

        # remove old parquet files
        os.system('rm -rf ./outfiles/' + job_name + ch)


if __name__ == "__main__":
    # e.g.
    # run locally as: python run.py --year 2017 --processor hww --starti 0 --endi 1 --pfnano=False --all=False

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',        dest='year',           default='2017',                     help="year",                                type=str)
    parser.add_argument('--starti',      dest='starti',         default=0,                          help="start index of files",                type=int)
    parser.add_argument('--endi',        dest='endi',           default=-1,                         help="end index of files",                  type=int)
    parser.add_argument("--processor",   dest="processor",      default="hww",                      help="HWW processor",                       type=str)
    parser.add_argument("--dask",        dest='dask',           default=False,                      help="Run with dask",                       action=BoolArg)
    parser.add_argument('--samples',     dest='samples',        default="samples_config.json",      help='path to datafiles',                   type=str)
    parser.add_argument("--pfnano",      dest='pfnano',         default=False,                      help="Run with pfnano",                     action=BoolArg)
    parser.add_argument("--chunksize",   dest='chunksize',      default=10000,                      help="chunk size in processor",             type=int)
    parser.add_argument("--all",         dest='all',            default=True,                       help="Run over all samples in the config",  action=BoolArg)
    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    args = parser.parse_args()

    main(args)
