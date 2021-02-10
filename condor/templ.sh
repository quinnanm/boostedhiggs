#!/bin/sh

#get setup coffea area
xrdcp root://cmseos.fnal.gov//store/user/cmantill/coffeaenv.tgz ./coffeaenv.tgz
tar -zxf coffeaenv.tgz
source coffeaenv/bin/activate

# copy code
xrdcp -f root://cmseos.fnal.gov//store/user/cmantill/boostedhiggs.tar.gz ./boostedhiggs.tar.gz
tar -zxvf boostedhiggs.tar.gz 
xrdcp root://cmseos.fnal.gov/EOSDIR/SCRIPTNAME .
# copy other scripts needed

# make dir for input/output
mkdir infiles
xrdcp root://cmseos.fnal.gov/EOSINDIR/YEAR_SAMPLE.json infiles
mkdir outfiles

# run code
python SCRIPTNAME YEAR SAMPLE

#move output to eos
xrdcp -f outfiles/*.coffea EOSOUT
