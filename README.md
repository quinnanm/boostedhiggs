# boostedhiggs

## Quickstart

### Pre-requisites for cmslpc-sl7.fnal.gov:
```bash
# check your platform: CC7 shown below, for SL6 it would be "x86_64-slc6-gcc8-opt"
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.csh  # or .csh, etc.
# might need for lpc
pip install entrypoints==0.3.0 --user
```

### Pre-requisites (locally):
Install miniconda (if you do not already have it):
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the instructions to finish the installation
# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]

# disable auto activation of the base environment
conda config --set auto_activate_base false
```
Verify the installation is successful by running conda info and check if the paths are pointing to your Miniconda installation.

If you cannot run conda command, check if you need to add the conda path to your PATH variable in your bashrc/zshrc file, e.g.,
```
export PATH="$HOME/miniconda3/bin:$PATH"
```

Set up a conda environment and install the required packages:
```
# create a new conda environment
conda create -n hhbbww python=3.7

# activate the environment
conda activate hhbbww

# install coffea
pip install coffea==0.7.0

# jupyter if you do not have it
pip install jupyter
```

### Clone repo
```bash
git clone git@github.com:cmantill/boostedhiggs.git
cd boostedhiggs/
pip install --user --editable .
```

## Testing locally

## Location of ntuples (for 2017)
```
/eos/uscms/store/user/cmantill/PFNano/
```

Download a file and put it in `binder/data/`, e.g.:
```
scp cmslpc-sl7.fnal.gov:/eos/uscms/store/user/cmantill/PFNano/2017_preUL/BulkGravTohhTohVVhbb_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia8/RunIIFall17Jan22-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/210202_002316/0000/nano_mc2017_1.root binder/data/BulkGravTohhTohVVhbb_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia8_nano_mc2017_1.root
```

And then run the notebook:
https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/binder/bbwwprocessor.ipynb