# Submitting condor jobs

## Start singularity shell with `coffea` environment

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

## Condor setup

- Start your proxy

```
voms-proxy-init --voms cms --valid 168:00
```

- Change your username (and name of the file if needed) for the `x509userproxy` path in the condor template file:

- Change output directory and username in `submit.py`, e.g.:
```
homedir = '/store/user/$USER/boostedhiggs/'
```

## Submitting jobs

```
python condor/submit.py Aug17 run.py 5 2017
for i in condor/Aug17/*/*.jdl; do condor_submit $i; done
```

e.g.
```
condor_submit condor/Aug11/SingleMuon_2017_2.jdl
```