#!/bin/bash

python3 -m pip install correctionlib==2.1.0
pip install --upgrade numpy==1.21.5

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python run.py --year YEAR --processor input --pfnano v2_2 --n NUMJOBS --starti STARTI --sample SAMPLE --inference

# remove incomplete jobs
rm -rf outfiles/*mu
rm -rf outfiles/*ele

# # setup CMSSW
# xrdcp -f root://cmseos.fnal.gov//store/user/cmantill/CMSSW_11_1_0_pre5_PY3.tgz ./CMSSW_11_1_0_pre5_PY3.tgz
# tar -zxvf CMSSW_11_1_0_pre5_PY3.tgz
# rm *.tgz
# cd CMSSW_*/src
# scram b ProjectRename
# eval `scramv1 runtime -sh`
# cmsenv
# cd ../../

# # merge rootfiles
# hadd outfiles/job_name.root outfiles/outroot/job_name/*
# rm -r outfiles/outroot

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
