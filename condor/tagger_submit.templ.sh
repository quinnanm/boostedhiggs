#!/bin/bash

python3 -m pip install correctionlib==2.1.0
pip install --upgrade numpy==1.21.5

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --processor input --local --sample SAMPLE --n NUMJOBS --starti STARTI --inference --year YEAR
python run.py --processor input --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 4 --starti 0 --inference --year 2016APV


# remove incomplete jobs
rm -rf outfiles/*mu
rm -rf outfiles/*ele

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
