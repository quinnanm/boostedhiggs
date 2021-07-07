# binder notebooks

## Local setup:

### Install miniconda (if you don't already have it):
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

If you cannot run the conda command, check if you need to add the conda path to your PATH variable in your bashrc/zshrc file, e.g.,
```
export PATH="$HOME/miniconda3/bin:$PATH"
```

### Set up a conda environment and install the required packages
```
# create a new conda environment
conda create -n hww python=3.7

# activate the environment
conda activate hww

# install the necessary python packages (latest coffea version)
pip install numpy matplotlib coffea mplhep
```

## Copy input data files
An example data file (for HWW semi-lep signal - UL):
```
mkdir data/
scp -r cmslpc-sl7.fnal.gov:/eos/uscms/store/user/lpcdihiggsboost/cmantill/PFNano/2017_UL_ak15/GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8/RunIISummer19UL17Jun23-106X_mc2017_realistic_v6-v2/210623_225150/0000/nano_mc2017_1-130.root data/nano_mc2017_1-130.root
```

More files (to run e.g. on coffea-casa) can be found in the fileset json:
```
data/fileset/2017_UL_ak15.json 
```

