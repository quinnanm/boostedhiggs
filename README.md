# boostedhiggs

## Quickstart
```bash
# check your platform: CC7 shown below, for SL6 it would be "x86_64-slc6-gcc8-opt"
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.csh  # or .csh, etc.
git clone git@github.com:cmantill/boostedhiggs.git
cd boostedhiggs
# might need for lpc
pip install entrypoints==0.3.0 --user
pip install --user --editable .
```
## For condor processing
```bash
bash envSetup.sh
```

## Location of ntuples (for 2017)
```
/eos/uscms/store/user/cmantill/PFNano/
```
