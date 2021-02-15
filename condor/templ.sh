#!/bin/sh

# remove old
source /cvmfs/cms.cern.ch/cmsset_default.sh
rm *.tar.gz

# copy code                                                                                                                                                                                                
xrdcp -f root://cmseos.fnal.gov//store/user/cmantill/boostedhiggs.tar.gz ./boostedhiggs.tar.gz
tar -zxvf boostedhiggs.tar.gz
xrdcp root://cmseos.fnal.gov//store/user/cmantill/coffeaenv.tar.gz .
tar -zxf coffeaenv.tar.gz

# setup environment
source coffeaenv/bin/activate
export PYTHONPATH=${PWD}:${PYTHONPATH}

# print versions
python -c "import awkward; print(awkward.__version__)"
python -c "import coffea; print(coffea.__version__)"

# run code - needs to be run in this directory (because pythonpath starts w. ./coffeaenv)
cp boostedhiggs/run.py .
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE 

# move output to eos
xrdcp -f hhbbww.coffea EOSOUT

# remove stuff
rm *.tar.gz
rm *.coffea
rm *.py
rm *.sh
rm *.md
rm *.flake8
