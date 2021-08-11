#!/bin/bash

# make dir for output
mkdir outfiles

# run code (e.g. for run.py)
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE --processor PROCESSOR --condor

#move output to eos
xrdcp -f outfiles/*.hist EOSOUT
