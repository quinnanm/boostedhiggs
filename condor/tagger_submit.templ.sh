#!/bin/bash

python3 -m pip install correctionlib==2.1.0
pip install --upgrade numpy==1.21.5

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --processor input --pfnano v2_2 --inference --n NUMJOBS --starti STARTI --sample SAMPLE

# remove incomplete jobs
rm -rf outfiles/*mu
rm -rf outfiles/*ele

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
