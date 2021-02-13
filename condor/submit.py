#!/usr/bin/python

import argparse
import os
import re
import fileinput

import json
import glob

'''
 Submit condor jobs of processor
 Run as e.g.: python submit.py Feb14 run.py 20 1   
 Arguments:
  = [0]: tag of jobs and output dir in eos e.g. Feb14
  - [1]: script to run e.g. run.py (needs to be in boostedhiggs directory)
  - [2]: number of files per job e.g. 20
  - [3]: whether to re-tar the boostedhiggs directory (do this if the processor changed): 1 to re-tar
'''
# Note: change username in `cmantill` in this script

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('settings', metavar='S', type=str, nargs='+', help='label scriptname (re-tar)')
args = parser.parse_args()

if (not ((len(args.settings) is 3) or (len(args.settings) is 4))):
    print("Wrong number of arguments (must be 3 or 4, found",len(args.settings),")")
    sys.exit()

label = args.settings[0]
script = args.settings[1] # should be run.py
files_per_job = int(args.settings[2])

loc_base = os.environ['PWD']
logdir = label
homedir = '/store/user/cmantill/bbww/'+logdir+'/'
outdir = homedir + '/outfiles/'

# list of samples to run
samplelist = {
    # 'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017',
    # 'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017',
    # 'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017',
    # 'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017' 
    # 'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017', 
    # 'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': '2017', 
    # 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8': '2017',
    # 'GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8': '2017'
    # 'BulkGravTohhTohVVhbb_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia8': '2017', 
    # 'BulkGravTohhTohVVhbb_narrow_M-1400_TuneCP5_13TeV-madgraph-pythia8': '2017',
    # 'BulkGravTohhTohVVhbb_narrow_M-1800_TuneCP5_13TeV-madgraph-pythia8': '2017',
    # 'BulkGravTohhTohVVhbb_narrow_M-2000_TuneCP5_13TeV-madgraph-pythia8': '2017',
    # 'BulkGravTohhTohVVhbb_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia8': '2017',
    'BulkGravTohhTohVVhbb_narrow_M-4500_TuneCP5_13TeV-madgraph-pythia8': '2017',
}


# re-tar boostedhiggs files
os.chdir('../../')
if (len(args.settings) is 4):
    os.system('tar -vzcf boostedhiggs.tar.gz boostedhiggs/ --exclude="*.root" --exclude="*.pdf" --exclude="*.pyc" --exclude=tmp --exclude="*.tgz" --exclude="*std*" --exclude="*sum*" --exclude-vcs --exclude-caches-all --exclude="*.sh" --exclude=coffeaenv --exclude=condor --exclude=*.tar.gz')
    os.system('xrdcp -f boostedhiggs.tar.gz root://cmseos.fnal.gov//store/user/cmantill/boostedhiggs.tar.gz')
    os.system('rm boostedhiggs.tar.gz')
os.chdir(loc_base)

# name to give your output files
prefix = label

# make local directory
locdir = 'logs/'+logdir
os.system('mkdir -p  %s' %locdir)

# and condor directory
print('CONDOR work dir: '+outdir)
os.system('mkdir -p /eos/uscms'+outdir)

# read files from samples given
# TODO: add all years
totfiles = {}
with open('../data/fileset_2017preUL.json', 'r') as f:
    newfiles = json.load(f)
    totfiles.update(newfiles)
with open('../data/fileset_2017UL.json', 'r') as f:
    newfiles = json.load(f)
    totfiles.update(newfiles)
for sample in samplelist:
    totfiles[sample] = len(totfiles[sample])

# submit jobs
nsubmit = 0
for sample in samplelist:

    prefix = sample+'_'+samplelist[sample]
    print('Submitting '+prefix)

    njobs = int(totfiles[sample]/files_per_job)+1
    remainder = totfiles[sample]-int(files_per_job*(njobs-1))

    for j in range(njobs):

        condor_templ_file = open(loc_base+"/templ.jdl")
        sh_templ_file    = open(loc_base+"/templ.sh")
    
        localcondor = locdir+'/'+prefix+"_"+str(j)+".jdl"
        condor_file = open(localcondor,"w")
        for line in condor_templ_file:
            line=line.replace('DIRECTORY',locdir)
            line=line.replace('PREFIX',prefix)
            line=line.replace('JOBID',str(j))
            condor_file.write(line)
        condor_file.close()
    
        localsh=locdir+'/'+prefix+"_"+str(j)+".sh"
        eosoutput="root://cmseos.fnal.gov/"+outdir+"/"+prefix+'_'+str(j)+'.coffea'
        sh_file = open(localsh,"w")
        for line in sh_templ_file:
            line=line.replace('SCRIPTNAME',script)
            line=line.replace('FILENUM',str(j))
            line=line.replace('YEAR',samplelist[sample])
            line=line.replace('SAMPLE',sample)
            line=line.replace('STARTNUM',str(j*files_per_job))
            line=line.replace('ENDNUM',str((j+1)*files_per_job))
            line=line.replace('EOSOUT',eosoutput)
            sh_file.write(line)
        sh_file.close()

        os.system('chmod u+x '+locdir+'/'+prefix+'_'+str(j)+'.sh')
        if (os.path.exists('%s.log'  % localcondor)):
            os.system('rm %s.log' % localcondor)
        os.system('condor_submit %s' % localcondor)

        condor_templ_file.close()
        sh_templ_file.close()

        nsubmit = nsubmit + 1

print(nsubmit,"jobs submitted.")
