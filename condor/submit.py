#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantill, Raghav Kansal
"""
import json
import argparse
import os
from math import ceil


def main(args):
    locdir = "condor/" + args.tag
    homedir = f"/store/user/fmokhtar/boostedhiggs/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    dataset_name = "GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8"
    with open("binder/fileset_2017_UL_NANO.json", 'r') as f:
        files = json.load(f)[dataset_name]

    fileset = {}
    # need to define the fileset but call them with xcache
    fileset[dataset_name] = ["root://xcache/"+ f for f in files]
    
    # directories for every sample
    for sample in fileset:
        os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        print("Submitting " + sample)

        tot_files = len(fileset[sample])
        njobs = ceil(tot_files / args.files_per_job)

        for j in range(njobs):
            if args.test and j == 2:
                break
            condor_templ_file = open("src/condor/submit.templ.jdl")

            localcondor = f"{locdir}/{sample}_{j}.jdl"
            condor_file = open(localcondor, "w")
            for line in condor_templ_file:
                line = line.replace("DIRECTORY", locdir)
                line = line.replace("PREFIX", sample)
                line = line.replace("JOBID", str(j))
                condor_file.write(line)

            condor_file.close()
            condor_templ_file.close()

            sh_templ_file = open("src/condor/submit.templ.sh")

            localsh = f"{locdir}/{sample}_{j}.sh"
            eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"
            eosoutput_pkl = f"{eosoutput_dir}/out_{j}.pkl"
            sh_file = open(localsh, "w")
            for line in sh_templ_file:
                line = line.replace("SCRIPTNAME", args.script)
                line = line.replace("YEAR", args.year)
                line = line.replace("SAMPLE", sample)
                line = line.replace("PROCESSOR", args.processor)
                line = line.replace("STARTNUM", str(j * args.files_per_job))
                line = line.replace("ENDNUM", str((j + 1) * args.files_per_job))
                line = line.replace("EOSOUTDIR", eosoutput_dir)
                line = line.replace("EOSOUTPKL", eosoutput_pkl)
                sh_file.write(line)
            sh_file.close()
            sh_templ_file.close()

            os.system(f"chmod u+x {localsh}")
            if os.path.exists("%s.log" % localcondor):
                os.system("rm %s.log" % localcondor)

            print("To submit ", localcondor)
            # os.system('condor_submit %s' % localcondor)

            nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="run.py", help="script to run", type=str)
    parser.add_argument("--test", default=False, help="test run or not - test run means only 2 jobs per sample will be created", type=bool)
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument("--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str)
    parser.add_argument("--processor", dest="processor", default="hww", help="which processor", type=str, choices=["hww"])
    parser.add_argument("--samples", dest="samples", default=[], help="which samples to run, default will be all samples", nargs="*")
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    args = parser.parse_args()

    main(args)