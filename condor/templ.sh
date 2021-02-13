#!/bin/sh

#get setup coffea area
xrdcp root://cmseos.fnal.gov//store/user/cmantill/coffeaenv.tar.gz ./coffeaenv.tar.gz
tar -zxf coffeaenv.tar.gz
source coffeaenv/bin/activate
export PYTHONPATH=${PWD}:${PYTHONPATH}

# copy code
xrdcp -f root://cmseos.fnal.gov//store/user/cmantill/boostedhiggs.tar.gz ./boostedhiggs.tar.gz
tar -zxvf boostedhiggs.tar.gz 

# run code
mkdir test/
cp boostedhiggs/SCRIPTNAME test/
cd test/
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE 

#move output to eos
xrdcp -f hhbbww.coffea EOSOUT
