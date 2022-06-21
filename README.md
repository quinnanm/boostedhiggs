# boostedhiggs

<!-- TOC -->

- [boostedhiggs](#boostedhiggs)
    - [Set up your environment](#setting-up-coffea-environments)
        - [With conda](#with-conda)
        - [With singularity shell](#with-singularity-shell)
        - [With python environments](#with-python-environments)
    - [Structure of the repository](#structure-of-the-repository)
        - [Data fileset](#data-fileset)
    - [Submitting condor jobs!](#submitting-condor-jobs)
        - [First time setup](#first-time-setup)
        - [Submit](#submitting-jobs)
        - [Post-processing](#post-processing)

<!-- /TOC -->


## Setting up coffea environments

### With conda

#### Install miniconda (if you do not have it already)
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
pip install numpy pandas scikit-learn coffea correctionlib pyarrow

# install xrootd
conda install -c conda-forge xrootd
```

### With singularity shell

First, copy the executable script from `lpcjobqueue`: https://github.com/CoffeaTeam/lpcjobqueue
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

### Data fileset

The .json files containing the datasets to be run should be saved in the same `data/` directory.

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

### Submitting jobs
- Before submitting jobs, make sure you have a valid proxy:
```
voms-proxy-init --voms cms --valid 168:00
```

We use the `submit.py` script to submit jobs.

For example:
```
python condor/submit.py --year 2017 --tag Feb21 --samples samples_pfnano.json --pfnano
```
where:
- year: this determines which fileset to read
- tag: is a tag to the jobs (usually a date or something more descriptive)
- samples: a json file that contains the names of the samples to run and the number of files per job for that sample
--pfnano: use pfnano
--no-pfnano: do not use pfnano
- number of files per job: if given all of the samples will use these number of files per job
- script that runs processor: is `run.py` by default

e.g.
```
python3 condor/submit.py --year 2017 --tag Jun20 --samples samples_pfnano_mc.json --pfnano --slist ggHToWWTo4Q-MH125,GluGluHToWWTo4q,GluGluHToWWTo4q-HpT190,TTToSemiLeptonic --submit
```

The `run.py` script has different options to e.g. select a different processor, run over files that go from one starting index (starti) to the end (endi).

The `submit.py` creates the submission files **and submits jobs afterwards if --submit is True.**

To submit jobs one does:
```
for i in condor/${TAG}/*.jdl; do condor_submit $i; done
```
or one can individually submit jobs.

You can check the status of your jobs with:
```
condor_q
```
If you see no jobs listed it means they have all finished.

#### Testing jobs locally per single sample:
```
python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --sample GluGluHToWWToLNuQQ --local
```

#### Testing jobs locally over multiple samples specified in the json:
```
python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --json samples_pfnano.json
```

Make parquets. Run postprocess_parquets.py to add a "tot_weight" column.
