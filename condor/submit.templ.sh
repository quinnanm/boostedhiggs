#!/bin/bash

jobid=$1

python3 -m pip install correctionlib==2.1.0
pip install --upgrade numpy==1.21.5

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --processor PROCESSOR PFNANO INFERENCE SELECTION --n NUMJOBS --starti ${jobid} --sample SAMPLE --config METADATAFILE --channels CHANNELS --label LABEL

# remove incomplete jobs
rm -rf outfiles/*mu
rm -rf outfiles/*ele

# in cmsenv
hadd outfiles/job_name.root outfiles/outroot/job_name/*
rm -r outfiles/outroot

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
