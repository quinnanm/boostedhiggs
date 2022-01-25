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

    with open("fileset_2017_UL_NANO.json", 'r') as f:
        fileset = json.load(f)
    
    locdir = "condor/" + args.tag
    homedir = f"/store/user/fmokhtar/boostedhiggs/"
    outdir = homedir + args.tag + "/"

    # make local directory
    logdir = locdir + "/logs"
    os.system(f"mkdir -p {logdir}")

    # and condor directory
    print("CONDOR work dir: " + outdir)
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    samples = [
        "GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8",
        # "QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8",
        # "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        # "TTToHadronic_TuneCP5_13TeV-powheg-pythia8",
        # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        # "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        # "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        # "ST_t-channel_muDecays_TuneCP5_13TeV-comphep-pythia8",
        # "ST_t-channel_eleDecays_TuneCP5_13TeV-comphep-pythia8",
        # "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
        # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
        # "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
        # "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
        # "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
        # "DYJetsToLL_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        # "DYJetsToLL_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        # "DYJetsToLL_Pt-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        # "DYJetsToLL_Pt-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        # "SingleElectron",
        # "SingleMuon",
    ]
    
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