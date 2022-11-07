#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantilla, Raghav Kansal, Farouk Mokhtar
"""
import json
import argparse
import os
from math import ceil
from condor.file_utils import loadJson


def main(args):
    
    username = os.environ["USER"]
    homedir = f"/store/user/{args.username}/boostedhiggs/"
    outdir = "/eos/uscms/" + homedir + args.tag + "_" + args.year + "/"

    # build metadata.json with samples
    slist = args.slist.split(',') if args.slist is not None else None

    # TODO: this args.samples should be the one copied over in the condor/ directory at the moment the job was submitted
    files, nfiles_per_job = loadJson(args.samples, args.year, args.pfnano, slist)

    # submit a cluster of jobs per sample
    for sample in files.keys():
        tot_files = len(files[sample])
        
        njobs = ceil(tot_files / nfiles_per_job[sample])
        files_per_job = str(nfiles_per_job[sample])

        print(f"Sample {sample} should produce {njobs} files")

        import glob
        njobs_produced = len(glob.glob1(f"{outdir}/{sample}/outfiles","*.pkl"))
        print(f"Sample {sample} produced {njobs_produced} files")

        if njobs_produced!=njobs:   # debug which pkl file wasn't produced
            print(f"-----> SAMPLE HAS RAN INTO ERROR")
            for i in range(0, njobs*nfiles_per_job[sample], nfiles_per_job[sample]):
                fname = f"{i}-{i+nfiles_per_job[sample]}"
                if not os.path.exists(f"{outdir}/{sample}/outfiles/{fname}.pkl"):
                    print(f"file {fname}.pkl wasn't produced..")
        print("-----------------------------------------------------------------------------------------")


if __name__ == "__main__":
    """
    python xplore_failed_jobs.py --pfnano --year 2017 --username cmantill --tag Nov4 --samples samples_pfnano_mc.json 
    python xplore_failed_jobs.py --pfnano --year 2017 --username fmokhtar --tag lumi --samples samples_pfnano_data.json 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--year",      dest="year",      default="2017",                help="year", type=str)
    parser.add_argument("--username",  dest="username",  default="cmantill",            help="user who submitted the jobs", type=str)
    parser.add_argument("--tag",       dest="tag",       default="Test",                help="process tag", type=str)
    parser.add_argument('--samples',   dest='samples',   default="samples_pfnano.json", help='path to datafiles', type=str)
    parser.add_argument('--slist',     dest='slist',     default=None,
                        help="give sample list separated by commas")
    parser.add_argument("--pfnano",    dest='pfnano', action='store_true')
    parser.add_argument("--no-pfnano", dest='pfnano', action='store_false')
    parser.set_defaults(pfnano=True)
    parser.set_defaults(inference=True)
    args = parser.parse_args()

    main(args)
