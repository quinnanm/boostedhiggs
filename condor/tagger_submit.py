#!/usr/bin/python

"""
Splits the total fileset and creates condor job submission files for the specified run script.
Author(s): Cristina Mantilla, Raghav Kansal, Farouk Mokhtar
"""
import argparse
import json
import os


def main(args):
    try:
        proxy = os.environ["X509_USER_PROXY"]
    except ValueError:
        print("No valid proxy. Exiting.")
        exit(1)

    locdir = "condor/" + args.tag + "_" + args.year
    username = os.environ["USER"]
    outdir = f"/store/user/{username}/boostedhiggs/" + args.tag + "_" + args.year + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    metadata_file = "metadata_mc.json"
    with open(f"{locdir}/{metadata_file}", "w") as f:
        json.dump(2, f, sort_keys=True, indent=2)

    # submit a cluster of jobs per sample
    sample = args.sample

    print(f"Making directory /eos/uscms/{outdir}/{sample}")
    os.system(f"mkdir -p /eos/uscms/{outdir}/{sample}")

    localcondor = f"{locdir}/{sample}.jdl"
    localsh = f"{locdir}/{sample}.sh"
    try:
        os.remove(localcondor)
        os.remove(localsh)
        os.remove(f"{locdir}/*.log")
    except Exception:
        pass

    # make submit.txt with number of jobs = 1
    jobids = [str(jobid) for jobid in range(1)]
    jobids_file = os.path.join(locdir, f"submit_{sample}.txt")
    with open(jobids_file, "w") as f:
        f.write("\n".join(jobids))

    # make condor file
    condor_templ_file = open("condor/tagger_submit.templ.jdl")
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
    sh_templ_file = open("condor/tagger_submit.templ.sh")
    eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{sample}/"
    eosoutput_pkl = f"{eosoutput_dir}/"
    sh_file = open(localsh, "w")
    for line in sh_templ_file:
        line = line.replace("SCRIPTNAME", args.script)
        line = line.replace("YEAR", args.year)
        line = line.replace("METADATAFILE", metadata_file)
        line = line.replace("NUMJOBS", str(args.n))
        line = line.replace("STARTI", str(args.starti))
        line = line.replace("SAMPLE", sample)
        line = line.replace("EOSOUTPKL", eosoutput_pkl)
        sh_file.write(line)
    sh_file.close()
    sh_templ_file.close()

    os.system(f"chmod u+x {localsh}")
    if os.path.exists("%s.log" % localcondor):
        os.system("rm %s.log" % localcondor)

    # submit
    if args.submit:
        print("Submit ", localcondor)
        os.system("condor_submit %s" % localcondor)


if __name__ == "__main__":
    """
    python condor/tagger_submit.py --year 2018 --tag test --sample TTToSemiLeptonic --n 100 --starti 0
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", dest="script", default="run.py", help="script to run", type=str)
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument("--submit", dest="submit", action="store_true", help="submit jobs when created")
    parser.add_argument("--n", default=None, help="# files per condor job", type=int)
    parser.add_argument("--starti", default=None, help="index to start from", type=int)
    parser.add_argument("--sample", dest="sample", default=None, help="specify sample", type=str)

    args = parser.parse_args()

    main(args)
