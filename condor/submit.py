import argparse
import os
import re
import fileinput

import json
import glob
import sys

'''
 Submit condor jobs of processor
 Run as e.g.: python submit.py Aug15 run.py 20 2017
 Arguments:
  = [0]: tag of jobs and output dir in eos e.g. Jul1
  - [1]: script to run e.g. run.py (needs to be included in transfer_files in templ.jdl)
  - [2]: number of files per job e.g. 20
  - [3]: year
'''
# Note: change username in `cmantill` in this script

homedir = "/store/user/fmokhtar/boostedhiggs/"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('settings', metavar='S', type=str, nargs='+', help='label scriptname (re-tar)')
args = parser.parse_args()

if (not ((len(args.settings) == 3) or (len(args.settings) == 4))):
    print("Wrong number of arguments (must be 3 or 4, found", len(args.settings), ")")
    sys.exit()

label = args.settings[0]
script = args.settings[1]  # should be run.py
files_per_job = int(args.settings[2])
year = args.settings[3]

loc_base = os.environ['PWD']

# list of samples to run and recos to use
recos = ["UL"]  # ,"preUL"]

# if empty run over all the datasets in both filesets
samples = {
    "UL": [
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
    ],
    "preUL": [],
}

# load files from datasets
for reco in recos:
    locdir = 'condor/' + label + '/' + reco
    logdir = locdir + '/logs/'
    os.system('mkdir -p  %s' % logdir)

    outdir = homedir + label + '_' + reco + '/outfiles/'
    os.system('mkdir -p  /eos/uscms/%s' % outdir)

    totfiles = {}
    with open('fileset/fileset_%s_%s_NANO.json' % (year, reco), 'r') as f:
        newfiles = json.load(f)
        totfiles.update(newfiles)

    samplelist = []
    if reco in samples.keys():
        samplelist = samples[reco]
    else:
        samplelist = totfiles.keys()

    # write json to metadata.json
    with open("%s/metadata.json" % (locdir), 'w') as json_file:
        json.dump(totfiles, json_file, indent=4, sort_keys=True)

    # copy script to locdir
    os.system('cp %s %s' % (script, locdir))

    # submit jobs
    nsubmit = 0
    for sample in samplelist:
        prefix = sample + '_%s_%s' % (year, reco)
        print('Submitting ' + prefix)

        njobs = int(len(totfiles[sample]) / files_per_job) + 1

        for j in range(njobs - 1):
            condor_templ_file = open(loc_base + "/condor/submit.templ.jdl")
            sh_templ_file = open(loc_base + "/condor/submit.templ.sh")

            localcondor = '%s/%s_%i.jdl' % (locdir, prefix, j)
            condor_file = open(localcondor, "w")
            for line in condor_templ_file:
                line = line.replace('DIRECTORY', locdir)
                line = line.replace('PREFIX', prefix)
                line = line.replace('JOBID', str(j))
                line = line.replace('JSON', "%s/metadata.json" % (locdir))
                condor_file.write(line)
            condor_file.close()

            localsh = '%s/%s_%i.sh' % (locdir, prefix, j)
            eosoutput = 'root://cmseos.fnal.gov/%s/%s_%i.hist' % (outdir, prefix, j)
            sh_file = open(localsh, "w")
            for line in sh_templ_file:
                line = line.replace('SCRIPTNAME', script)
                line = line.replace('FILENUM', str(j))
                line = line.replace('YEAR', year)
                line = line.replace('SAMPLE', sample)
                line = line.replace('PROCESSOR', 'hww')
                line = line.replace('STARTNUM', str(j * files_per_job))
                line = line.replace('ENDNUM', str((j + 1) * files_per_job))
                line = line.replace('EOSOUT', eosoutput)
                line = line.replace('OUTDIR', outdir)
                sh_file.write(line)
            sh_file.close()

            os.system('chmod u+x %s' % localsh)
            if (os.path.exists('%s.log' % localcondor)):
                os.system('rm %s.log' % localcondor)
            condor_templ_file.close()
            sh_templ_file.close()

            print('To submit ', localcondor)
            os.system('condor_submit %s' % localcondor)

            nsubmit = nsubmit + 1

    print(nsubmit, "jobs submitted.")
