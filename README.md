# boostedhiggs

## Setting up coffea environments

### With anaconda

#### Install miniconda if you do not have it already
Preferably, in your `nobackup` area (in LPC) or in your local computer:
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
#### Set up a conda environment and install the required packages
```
# create a new conda environment
conda create -n coffea-env python=3.7

# activate the environment
conda activate coffea-env

# install packages
pip install numpy pandas scikit-learn coffea correctionlib

# install xrootd
conda install -c conda-forge xrootd
```

### With singularity shell

First, copy the executable script from `lpcjobqueue`.
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

This creates two new files in this directory: `shell` and `.bashrc`.
The `./shell` executable can then be used to start a singularity shell with a coffea environment.

In order to be able to write to eos add the following line (replacing $USER with your username):
```-B /eos/uscms/store/user/$USER/boostedhiggs:/myeosdir```
in the shell executable.

e.g.
```
singularity exec -B ${PWD}:/srv -B /uscmst1b_scratch -B /eos/uscms/store/user/cmantill/boostedhiggs:/myeosdir --pwd /srv \
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE} \
  /bin/bash --rcfile /srv/.bashrc
```

### With python environments
This option is not recommended as we have found	some problems with uproot.
```
# this script sets up the python environment, only run once (again if you have not done)
./setup.sh

# this enables the environment, run it each login (csh users: use activate.csh)
source coffeaenv/bin/activate
```

## Structure of the repository

The main selection and histograms are defined in `boostedhiggs/hwwprocessor.py`.

The corrections are imported from `boostedhiggs/corrections.py`

The fileset json files that contain a dictionary of the files per sample are in the `fileset` directory.

<details><summary>Re-making the input dataset files with DAS</summary>
<p> 
  
```bash
# connect to LPC with a port forward to access the jupyter notebook server
ssh USERNAME@cmslpc-sl7.fnal.gov -L8xxx:localhost:8xxx

# create a working directory and clone the repo (if you have not done yet)
git clone git@github.com:cmantill/boostedhiggs/
cd boostedhiggs/

# enable the coffea environment, either the python environment
source coffeaenv/bin/activate

# or the conda 
conda activate coffea-env

# then activate your proxy
voms-proxy-init --voms cms --valid 100:00

# the json files are in the data directory
cd data/
jupyter notebook --no-browser --port 8xxx
```
There should be a link looking like `http://localhost:8xxx/?token=...`, displayed in the output at this point, paste that into your browser. 
You should see a jupyter notebook with a directory listing.
Open `filesetDAS.ipynb`.

The .json files containing the datasets to be run should be saved in the same `data/` directory.

</p>
</details>

## Submitting condor jobs

The condor setup uses the coffea singularity so make sure you have setup an script following the steps above.

### First time setup

- Change the singularity shell executable to have your eos directory. 
In order to be able to write to eos with condor jobs add the following line (replacing $USER with your username):
```-B /eos/uscms/store/user/$USER/boostedhiggs:/myeosdir```
in the shell executable.

e.g.
```
singularity exec -B ${PWD}:/srv -B /uscmst1b_scratch -B /eos/uscms/store/user/cmantill/boostedhiggs:/myeosdir --pwd /srv \
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE} \
  /bin/bash --rcfile /srv/.bashrc
```
- Change your username (and name of the file if needed) for the `x509userproxy` path in the condor template file:

- Change output directory and username in `submit.py`, e.g.:
```
homedir = '/store/user/$USER/boostedhiggs/'

### Submitting jobs
- Before submitting jobs, make sure you have a valid proxy:
```
voms-proxy-init --voms cms --valid 168:00
```

We use the `submit.py` script to submit jobs. 

The arguments are:
```
python condor/submit.py $TAG $SCRIPT_THAT_RUNS_PROCESSOR $NUMBER_OF_FILES_PER_JOB $YEAR
```
where:
- tag: is a tag to the jobs (usually a date or something more descriptive)
- script that runs processor: is `run.py` or similar. This script runs the coffea processor.
- number of files per job: is usually 1
- year: this determines which fileset to read.

The `run.py` script has different options to e.g. select a different processor, run over files that go from one starting index (starti) to the end (endi) in the `metadata.json` file.

If the `submit.py` script does not submit jobs by default (for testing purposes), one can do e.g:
```
python condor/submit.py Aug17 run.py 5 2017
for i in condor/Aug17/*/*.jdl; do condor_submit $i; done
```
or one can individually submit jobs with:
```
condor_submit condor/Aug17/SingleMuon_2017_2.jdl
```

## Postprocessing of jobs

