#!/bin/sh

#get setup coffea area
xrdcp root://cmseos.fnal.gov//store/user/cmantill/coffeaenv.tar.gz ./coffeaenv.tar.gz
tar -zxf coffeaenv.tar.gz
source coffeaenv/bin/activate
export PYTHONPATH=${PWD}:${PYTHONPATH}

python -c "import awkward; print(awkward.__version__)"

# copy code
xrdcp -f root://cmseos.fnal.gov//store/user/cmantill/boostedhiggs.tar.gz ./boostedhiggs.tar.gz
tar -zxvf boostedhiggs.tar.gz 

# run code
cd boostedhiggs/
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE 

#move output to eos
xrdcp -f hhbbww.coffea EOSOUT
