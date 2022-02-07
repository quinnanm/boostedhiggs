#!/bin/bash

python3 -m pip install correctionlib==2.0.0rc6

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --processor PROCESSOR --sample SAMPLE

# remove incomplete jobs
rm -rf outfiles/*had
rm -rf outfiles/*mu
rm -rf outfiles/*ele

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
