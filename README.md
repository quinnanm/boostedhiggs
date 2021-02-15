# boostedhiggs

## Running Locally

### Pre-requisites
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

## Condor setup (in bash shell)

Setup environment:
```
source setup.sh
```

Copy to eos
```
cp coffeaenv.tar.gz  /eos/uscms/store/user/cmantill/coffeaenv.tar.gz
```

Replace username in submit script

# To remake dataset input files
```
# connect to LPC with a port forward to access the jupyter notebook server
ssh USERNAME@cmslpc-sl7.fnal.gov -L8xxx:localhost:8xxx

# create a working directory and clone the repo (if have not done yet)
cd nobackup/hww/
git clone git@github.com:cmantill/boostedhiggs/
cd boostedhiggs/condor/

# this script sets up the python environment, only run once
./setup.sh

# this enables the environment, run it each login (csh users: use activate.csh)
source egammaenv/bin/activate

# this gives you permission to read CMS data via xrootd
voms-proxy-init --voms cms --valid 100:00

# in case you do not already have this in your .bashrc (or equivalent) please run
source /cvmfs/cms.cern.ch/cmsset_default.sh

jupyter notebook --no-browser --port 8xxx
```
There should be a link like `http://localhost:8xxx/?token=...` displayed in the output at this point, paste that into your browser. You should see a jupyter notebook with a directory listing.
Open `filesetDAS.ipynb`.

The .json files containing the datasets to be run should be saved in `infiles/`.

```
python run.py --year 2017 --starti 0 --endi 1 --samples BulkGravTohhTohVVhbb_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia8
```