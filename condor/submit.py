#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantill, Raghav Kansal
"""
import json
import argparse
import os
from math import ceil
from BoolArg import BoolArg


def main(args):

    locdir = "condor/" + args.tag
    homedir = "/store/user/fmokhtar/boostedhiggs/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

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

    fileset = {}

    with open(fname, 'r') as f:
        if args.pfnano:
            files = json.load(f)[args.year]
            for sample in samples:
                for subdir in files:
                    for key, flist in files[subdir].items():
                        if sample == key:
                            fileset[sample] = files[subdir][key]
        else:
            files = json.load(f)
            for sample in samples:
                fileset[sample] = files[sample]

    # directories for every sample
    for sample in samples:
        os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

    # submit jobs
    nsubmit = 0
    for sample in samples:
        print("Submitting " + sample)

        tot_files = len(fileset[sample])
        njobs = ceil(tot_files / args.files_per_job)

        for j in range(njobs):
            if args.test and j == 2:
                break
            condor_templ_file = open("condor/submit.templ.jdl")

            localcondor = f"{locdir}/{sample}_{j}.jdl"
            condor_file = open(localcondor, "w")
            for line in condor_templ_file:
                line = line.replace("DIRECTORY", locdir)
                line = line.replace("PREFIX", sample)
                line = line.replace("JOBID", str(j))
                condor_file.write(line)

            condor_file.close()
            condor_templ_file.close()

            sh_templ_file = open("condor/submit.templ.sh")

            localsh = f"{locdir}/{sample}_{j}.sh"
            eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"

            if args.processor == 'hww':
                eosoutput_pkl = f"{eosoutput_dir}/"
            else:
                eosoutput_pkl = f"{eosoutput_dir}/out_{j}.pkl"

            sh_file = open(localsh, "w")
            for line in sh_templ_file:
                line = line.replace("SCRIPTNAME", args.script)
                line = line.replace("YEAR", args.year)
                line = line.replace("PROCESSOR", args.processor)
                line = line.replace("STARTNUM", str(j * args.files_per_job))
                line = line.replace("ENDNUM", str((j + 1) * args.files_per_job))
                line = line.replace("EOSOUTPKL", eosoutput_pkl)
                line = line.replace("SAMPLE", sample)
                if args.pfnano:
                    line = line.replace("PFNANO", "True")
                else:
                    line = line.replace("PFNANO", "False")
                sh_file.write(line)
            sh_file.close()
            sh_templ_file.close()

            os.system(f"chmod u+x {localsh}")
            if os.path.exists("%s.log" % localcondor):
                os.system("rm %s.log" % localcondor)

            print("To submit ", localcondor)
            if args.submit:
                os.system('condor_submit %s' % localcondor)

            nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script",    dest="script", default="run.py", help="script to run", type=str)
    parser.add_argument("--test",      dest="test", default=False, help="test run or not - test run means only 2 jobs per sample will be created", type=bool)
    parser.add_argument("--year",      dest="year", default="2017", help="year", type=str)
    parser.add_argument("--tag",       dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument("--outdir",    dest="outdir", default="outfiles", help="directory for output files", type=str)
    parser.add_argument("--processor", dest="processor", default="hww", help="which processor", type=str, choices=["hww"])
    parser.add_argument('--samples',     dest='samples',        default="samples_config.json",      help='path to datafiles',                   type=str)
    parser.add_argument("--pfnano",      dest='pfnano',         default=False,                      help="Run with pfnano",                     action=BoolArg)
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    parser.add_argument("--submit",      dest='submit',         default=False,                      help="submit jobs when created",                     action=BoolArg)

    args = parser.parse_args()

    main(args)
