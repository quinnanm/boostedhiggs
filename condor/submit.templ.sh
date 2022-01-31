#!/bin/bash

python3 -m pip install correctionlib==2.0.0rc6

# make dir for output (not really needed cz python script will make it)
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --processor PROCESSOR --sample SAMPLE

# the star denotes the 3 folders (1 for each channel)... this will remove the single parquet files
export STARTNUM = STARTNUM
export ENDNUM = ENDNUM
rm -r outfiles/*/$STARTNUM-$ENDNUM/

#move output to eos
xrdcp -r -f outfiles/ EOSOUTPKL
