#!/usr/bin/python

import argparse
import json
import os
import pickle as pkl
import time

import pandas as pd
import uproot
from coffea import nanoevents, processor

nanoevents.PFNanoAODSchema.warn_missing_crossrefs = False


def main(args):
    # make directory for output
    OUTPATH = "./outfiles"
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    # if --local is specified in args, process only the args.sample provided
    if args.macos:
        files = {}
        files[args.sample] = [f"rootfiles/{args.sample}/nano_mc2017_{i}.root" for i in range(3)]

    elif args.local:
        files = {}
        with open(f"fileset/pfnanoindex_{args.pfnano}_{args.year}.json", "r") as f:
            files_all = json.load(f)
            for subdir in files_all[args.year]:
                for key, flist in files_all[args.year][subdir].items():
                    if key in args.sample:
                        files[key] = ["root://cmseos.fnal.gov/" + f for f in flist]

    else:
        # get samples
        if "metadata" in args.config:
            with open(args.config, "r") as f:
                files = json.load(f)
        else:
            if not args.config or not args.configkey:
                raise Exception("No config or configkey provided for condor jobs")

            # hopefully this step is avoided in condor jobs that have metadata.json
            from condor.file_utils import loadFiles

            files, _ = loadFiles(args.config, args.configkey, args.year, args.pfnano, args.sample.split(","))

            # print(files)

    if not files:
        print("Did not find files.. Exiting.")
        exit(1)

    # build fileset with files to run per job
    fileset = {}
    starti = args.starti
    job_name = "/" + str(starti * args.n)
    if args.n != -1:
        job_name += "-" + str(args.starti * args.n + args.n)
    for sample, flist in files.items():
        if args.sample:
            if sample not in args.sample.split(","):
                continue
        if args.n != -1:
            fileset[sample] = flist[args.starti * args.n : args.starti * args.n + args.n]
        else:
            fileset[sample] = flist

    print(
        len(list(fileset.keys())),
        "Samples in fileset to be processed: ",
        list(fileset.keys()),
    )
    print(fileset)

    # define processor
    from boostedhiggs.inputprocessor import InputProcessor

    p = InputProcessor(num_jets=1, output_location=f"{OUTPATH}/{job_name}")

    tic = time.time()
    if args.executor == "dask":
        from coffea.nanoevents import NanoeventsSchemaPlugin
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
        executor=executor,
        savemetrics=True,
        schema=nanoevents.PFNanoAODSchema,
        chunksize=args.chunksize,
    )

    out, metrics = run(fileset, "Events", processor_instance=p)

    elapsed = time.time() - tic
    print(f"Metrics: {metrics}")
    print(f"Finished in {elapsed:.1f}s")

    # dump to pickle
    filehandler = open("./outfiles/" + job_name + ".pkl", "wb")
    pkl.dump(out, filehandler)
    filehandler.close()

    # merge parquet
    data = pd.read_parquet("./outfiles/" + job_name + "/parquet")
    data.to_parquet("./outfiles/" + job_name + ".parquet")
    # remove old parquet files
    os.system("rm -rf ./outfiles/" + job_name)


if __name__ == "__main__":
    # python run_input_skimmer.py --macos --processor input --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0
    # python run_input_skimmer.py --local --processor input --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--starti", dest="starti", default=0, help="start index of files", type=int)
    parser.add_argument("--n", dest="n", default=-1, help="number of files to process", type=int)
    parser.add_argument("--config", dest="config", default=None, help="path to datafiles", type=str)
    parser.add_argument("--key", dest="configkey", default=None, help="config key: [data, mc, ... ]", type=str)
    parser.add_argument("--sample", dest="sample", default=None, help="specify sample", type=str)
    parser.add_argument("--processor", dest="processor", required=True, help="processor", type=str)
    parser.add_argument("--chunksize", dest="chunksize", default=10000, help="chunk size in processor", type=int)
    parser.add_argument(
        "--executor",
        type=str,
        default="futures",
        choices=["futures", "iterative", "dask"],
        help="type of processor executor",
    )
    parser.add_argument(
        "--pfnano",
        dest="pfnano",
        type=str,
        default="v2_2",
        help="pfnano version",
    )
    parser.add_argument("--macos", dest="macos", action="store_true")
    parser.add_argument("--local", dest="local", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")
    parser.add_argument("--no-inference", dest="inference", action="store_false")
    parser.add_argument("--without_selection", dest="without_selection", action="store_true")

    parser.set_defaults(inference=False)
    args = parser.parse_args()

    main(args)
