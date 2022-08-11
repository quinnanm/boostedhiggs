#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantilla, Raghav Kansal, Farouk Mokhtar
"""
import json
import argparse
import os
from math import ceil
from file_utils import loadJson


def main(args):

    try:
        proxy = os.environ["X509_USER_PROXY"]
    except:
        print('No valid proxy. Exiting.')
        exit(1)

    locdir = "condor/" + args.tag + "_" + args.year
    username = os.environ["USER"]
    homedir = f"/store/user/{username}/boostedhiggs/"
    outdir = homedir + args.tag + "_" + args.year + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    # build metadata.json with samples
    slist = args.slist.split(',') if args.slist is not None else None
    files, nfiles_per_job = loadJson(args.samples, args.year, args.pfnano, slist)
    metadata_file = f"metadata_{args.samples}"
    with open(f"{locdir}/{metadata_file}", "w") as f:
        json.dump(files, f, sort_keys=True, indent=2)

    # submit a cluster of jobs per sample
    for sample in files.keys():
        os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

        localcondor = f"{locdir}/{sample}.jdl"
        localsh = f"{locdir}/{sample}.sh"
        try:
            os.remove(localcondor)
            os.remove(localsh)
            os.remove(f"{locdir}/*.log")
        except:
            pass

        tot_files = len(files[sample])
        if args.files_per_job:
            njobs = ceil(tot_files / args.files_per_job)
            files_per_job = str(args.files_per_job)
        else:
            njobs = ceil(tot_files / nfiles_per_job[sample])
            files_per_job = str(nfiles_per_job[sample])

        # make submit.txt with number of jobs
        if args.test:
            njobs = 1
        jobids = [str(jobid) for jobid in range(njobs)]
        jobids_file = os.path.join(locdir, f'submit_{sample}.txt')
        with open(jobids_file, 'w') as f:
            f.write('\n'.join(jobids))

        # make condor file
        condor_templ_file = open("condor/submit.templ.jdl")
        condor_file = open(localcondor, "w")
        for line in condor_templ_file:
            line = line.replace("DIRECTORY", locdir)
            line = line.replace("PREFIX", sample)
            line = line.replace("JOBIDS_FILE", jobids_file)
            line = line.replace("METADATAFILE", metadata_file)
            line = line.replace("PROXY", proxy)
            condor_file.write(line)
        condor_file.close()
        condor_templ_file.close()

        # make executable file
        sh_templ_file = open("condor/submit.templ.sh")
        eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"
        eosoutput_pkl = f"{eosoutput_dir}/"
        sh_file = open(localsh, "w")
        for line in sh_templ_file:
            line = line.replace("SCRIPTNAME", args.script)
            line = line.replace("YEAR", args.year)
            line = line.replace("PROCESSOR", args.processor)
            line = line.replace("METADATAFILE", metadata_file)
            line = line.replace("NUMJOBS", files_per_job)
            line = line.replace("SAMPLE", sample)
            line = line.replace("EOSOUTPKL", eosoutput_pkl)
            if args.pfnano:
                line = line.replace("PFNANO", "--pfnano")
            else:
                line = line.replace("PFNANO", "--no-pfnano")
            if args.inference:
                line = line.replace("INFERENCE", "--inference")
            else:
                line = line.replace("INFERENCE", "--no-inference")
            sh_file.write(line)
        sh_file.close()
        sh_templ_file.close()

        os.system(f"chmod u+x {localsh}")
        if os.path.exists("%s.log" % localcondor):
            os.system("rm %s.log" % localcondor)

        # submit
        if args.submit:
            print('Submit ', localcondor)
            os.system('condor_submit %s' % localcondor)


if __name__ == "__main__":
    """
    python condor/submit.py --year 2017 --tag Aug11 --samples samples_pfnano_mc.json --pfnano --submit --inference
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--script",    dest="script",    default="run.py",              help="script to run", type=str)
    parser.add_argument("--year",      dest="year",      default="2017",                help="year", type=str)
    parser.add_argument("--tag",       dest="tag",       default="Test",                help="process tag", type=str)
    parser.add_argument("--processor", dest="processor", default="hww",                 help="which processor", type=str, choices=["hww"])
    parser.add_argument('--samples',   dest='samples',   default="samples_pfnano.json", help='path to datafiles', type=str)
    parser.add_argument('--slist',     dest='slist',     default=None,                  help="give sample list separated by commas")
    parser.add_argument("--test",      dest="test",      action='store_true',           help="only 2 jobs per sample will be created")
    parser.add_argument("--submit",    dest='submit',    action='store_true',           help="submit jobs when created")
    parser.add_argument("--files-per-job",               default=None,                  help="# files per condor job", type=int)
    parser.add_argument("--pfnano",    dest='pfnano', action='store_true')
    parser.add_argument("--no-pfnano", dest='pfnano', action='store_false')
    parser.add_argument("--inference",   dest='inference', action='store_true')
    parser.add_argument("--no-inference", dest='inference', action='store_false')
    parser.set_defaults(pfnano=True)
    parser.set_defaults(inference=True)
    args = parser.parse_args()

    main(args)
