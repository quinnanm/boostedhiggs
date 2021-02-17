# boostedhiggs

## Location of ntuples
For 2017 now:
```
/eos/uscms/store/user/cmantill/PFNano/
```

## Dataset input dictionary
We create a dictionary with all the datasets and files. 
The current ones are here: 
- [fileset 2017UL](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/data/fileset_2017UL.json)
- [fileset 2017](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/data/fileset_2017preUL.json)
- [fileset 2017 das](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/data/fileset_2017_das.json)

The first two have been produced by running this simple script that loops over files in eos [fileset_eos.py](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/data/fileset_eos.py).
While the latter has been produced with [fileset_das.ipyb](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/data/fileset_das.ipynb)

<details><summary>Re-making the input dataset files with DAS</summary>
<p> 
  
```bash
# connect to LPC with a port forward to access the jupyter notebook server
ssh USERNAME@cmslpc-sl7.fnal.gov -L8xxx:localhost:8xxx

# create a working directory and clone the repo (if you have not done yet)
cd nobackup/hww/
git clone git@github.com:cmantill/boostedhiggs/
git fetch origin
git checkout -b hhbbww_1l origin/hhbbww_1l
cd boostedhiggs/

# this script sets up the python environment, only run once (again if you have not done)
./setup.sh

# this enables the environment, run it each login (csh users: use activate.csh)
source coffeaenv/bin/activate

# this gives you permission to read CMS data via xrootd
voms-proxy-init --voms cms --valid 100:00

# in case you do not already have this in your .bashrc (or equivalent) please run
source /cvmfs/cms.cern.ch/cmsset_default.sh

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

## Running Locally
This is my favorite way to test producers (in your local computer).

### Pre-requisites
Install miniconda (if you do not already have it):
```bash
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

And then clone repo and branch:
```bash
git clone git@github.com:cmantill/boostedhiggs.git
cd boostedhiggs/
git fetch origin
git checkout -b hhbbww_1l origin/hhbbww_1l
pip install --user --editable .
```

### Testing locally
Download a file and put it in `binder/data/`, e.g.:
```
scp cmslpc-sl7.fnal.gov:/eos/uscms/store/user/cmantill/PFNano/2017_preUL/BulkGravTohhTohVVhbb_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia8/RunIIFall17Jan22-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/210202_002316/0000/nano_mc2017_1.root binder/data/BulkGravTohhTohVVhbb_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia8_nano_mc2017_1.root
```

The main coffea processor is [bbwwprocessor](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/boostedhiggs/bbwwprocessor.py).

One way to run it locally is via this [python notebook](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/binder/bbwwprocessor.ipynb). 
It runs over the file you just downloaded. Feel free to modify it to run over more or different files.


## Setup in LPC (bash shell)

For the first time:

```bash
ssh USERNAME@cmslpc-sl7.fnal.gov

# Create a working directory and clone the repo:
cd nobackup/hww/
git clone git@github.com:cmantill/boostedhiggs/
git fetch origin
git checkout -b hhbbww_1l origin/hhbbww_1l

# Set up the python environment, only run once (if you have not done before).
./setup.sh

# Copy the tarball of the python environment to your eos area, e.g.:
cp coffeaenv.tar.gz /eos/uscms/store/user/cmantill/coffeaenv.tar.gz
```

Now you should enable the environment, run it each login (csh users: use activate.csh)
```
source coffeaenv/bin/activate
```

It is also a good idea to initiate your proxy
```
voms-proxy-init --voms cms --valid 100:00

# in case you do not already have this in your .bashrc (or equivalent) please run
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

### Running in cmslpc nobackup

The main script that will run our processor is [run.py](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/run.py).
To check if it works you can do (in your coffeaenv environment):
```
python run.py --year 2017 --starti 0 --endi 1 --samples BulkGravTohhTohVVhbb_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia8
```

This should give you one `.coffea` output.

### Runnning jobs in condor

The first thing to do is to replace your username in: 
- [submit.py](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/condor/submit.py): this script submits the condor jobs
- [temp.sh](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/condor/templ.sh): this script is the bash template for the condor job

Next, you edit the datasets to run in [submit.py#L40](https://github.com/cmantill/boostedhiggs/blob/hhbbww_1l/condor/submit.py#L40).

To submit, do:
```
python submit.py LABEL run.py NUMFILESPERJOB XX
```
where:
- `LABEL`: is your job folder label e.g. a date or what you are currently testing. This will be the name of the folder in eos containing your output files.
- `NUMFILESPERJOB`: is the number of files per job e.g. 20
- `XX`: this is optional, if a number is present (e.g. 1) it will re-tar the boostedhiggs directory. This is necessary to do if your processor has -changed. So most likely you will have to include this option. 

The output of your jobs will be stored in:
```
/store/user/USERNAME/bbww/LABEL/
```
