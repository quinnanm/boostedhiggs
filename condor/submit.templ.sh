#!/bin/bash

python3 -m pip install correctionlib==2.0.0rc6

# make dir for output
mkdir outfiles

# run code (e.g. for run.py)
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --sample SAMPLE --processor PROCESSOR --condor --fileset metadata.json 

#move output to eos
xrdcp -f outfiles/*.hist EOSOUT
