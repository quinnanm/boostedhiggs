# Boosted Higgs To WW

- [Boosted Higgs To WW](#boosted-higgs-to-ww)
  * [Repository](#repository)
    + [Data fileset](#data-fileset)
  * [Submitting condor jobs](#submitting-condor-jobs)
    + [First time setup](#first-time-setup)
    + [Submitting jobs](#submitting-jobs)
      - [Testing jobs locally per single sample](#testing-jobs-locally-per-single-sample-)
      - [Testing jobs with inference (and triton server running)](#testing-jobs-with-inference--and-triton-server-running--)
      - [Testing jobs locally over multiple samples specified in the json](#testing-jobs-locally-over-multiple-samples-specified-in-the-json-)
  * [Triton server setup](#triton-server-setup)
    + [Running the server](#running-the-server)
    + [First time setup with a new model](#first-time-setup-with-a-new-model)
  * [Analysis](#analysis)
  * [Setting up coffea environments](#setting-up-coffea-environments)
    + [With conda](#with-conda)
      - [Install miniconda (if you do not have it already)](#install-miniconda--if-you-do-not-have-it-already-)
      - [Set up a conda environment and install the required packages](#set-up-a-conda-environment-and-install-the-required-packages)
    + [With singularity shell](#with-singularity-shell)

## Repository

### Data fileset

The .json files containing the datasets to be run should be saved in the same `data/` directory.

To update the fileset:
```
cd fileset/
python3 indexpfnano.py
```

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
python condor/submit.py --year 2017 --tag ${TAG} --samples samples_pfnano_mc.json --pfnano
```
where:
- year: this determines which fileset to read
- tag: is a tag to the jobs (usually a date or something more descriptive)
- samples: a json file that contains the names of the samples to run and the number of files per job for that sample
--pfnano: use pfnano
--no-pfnano: do not use pfnano
- number of files per job: if given all of the samples will use these number of files per job
- script that runs processor: is `run.py` by default
--no-inference: do not use inference
--inference: (true by default)

e.g.
```
python3 condor/submit.py --year 2017 --tag ${TAG} --samples samples_pfnano_mc.json --pfnano --slist GluGluHToWW_Pt-200ToInf_M-125,TTToSemiLeptonic --submit --no-inference
```

The `run.py` script has different options to e.g. select a different processor, run over files that go from one starting index (starti) to the end (endi).
By default `inference` is set in the run script.

The `submit.py` creates the submission files **and submits jobs afterwards if --submit is True.**

If `--submit` is not True, to submit jobs one can do:
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
python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local
```

#### Testing jobs with inference (and triton server running):
```
python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --sample GluGluHToWWToLNuQQ --local --inference
```

#### Testing jobs locally over multiple samples specified in the json:
```
python run.py --year 2017 --processor hww --pfnano --n 1 --starti 0 --json samples_pfnano_mc.json
```

## Triton server setup

### Running the server

To start triton server with kubernetes in PRP:
- Clone [gitlab repo](https://gitlab.nrp-nautilus.io/raghsthebest/triton-server).
- Change kubernetes namespace to triton:
  ```
  kubectl config set-context --current --namespace=triton
  ```
- Start the server:
  - For simple testing runs (with 2 gpus):
    ```
    kubectl create -f triton-inference-server-init.yaml -n triton
    ```
  - For scaling up (3x 2 gpus):
    ```
    kubectl create -f triton-inference-server-init-replicas.yaml -n triton
    ```
- Check that things are running:
  - Get pods (check that they are running - at least 2..):
    ```
    kubectl get pods  
    ```
    e.g. 
    ```
    % kubectl get pods
    NAME                      READY   STATUS    RESTARTS   AGE
    triton-588b6654bc-4j4hj   1/1     Running   0          134m
    triton-588b6654bc-jhn29   1/1     Running   0          134m
    triton-588b6654bc-z4lfp   1/1     Running   0          134m
    ```
  - Get logs
    ```
    kubectl logs triton-588b6654bc-4j4hj
    ```
    Wait until you see:
    ```
    ...
    I0808 11:16:50.401047 1 grpc_server.cc:225] Ready for RPC 'RepositoryIndex', 0
    I0808 11:16:50.401066 1 grpc_server.cc:225] Ready for RPC 'RepositoryModelLoad', 0
    I0808 11:16:50.401088 1 grpc_server.cc:225] Ready for RPC 'RepositoryModelUnload', 0
    I0808 11:16:50.401134 1 grpc_server.cc:416] Thread started for CommonHandler
    I0808 11:16:50.401474 1 grpc_server.cc:3144] New request handler for ModelInferHandler, 1
    I0808 11:16:50.401520 1 grpc_server.cc:2202] Thread started for ModelInferHandler
    I0808 11:16:50.401787 1 grpc_server.cc:3497] New request handler for ModelStreamInferHandler, 3
    I0808 11:16:50.401834 1 grpc_server.cc:2202] Thread started for ModelStreamInferHandler
    I0808 11:16:50.401858 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
    I0808 11:16:50.402790 1 http_server.cc:2795] Started HTTPService at 0.0.0.0:8000
    I0808 11:16:50.446070 1 sagemaker_server.cc:134] Started Sagemaker HTTPService at 0.0.0.0:8080
    I0808 11:16:50.488773 1 http_server.cc:162] Started Metrics Service at 0.0.0.0:8002
    ```
- IMPORTANT: Delete deployments when you are done:
  ```
  kubectl delete deployments triton -n triton
  ```

### First time setup with a new model

- Create a PR to this [repo](https://github.com/rkansal47/sonic-models/tree/master/models/) with the specific jitted model and with the updated config and labels.
  - Get the latest `state.pt` for the best epoch.
  - Use `make_jittable.py` in `weaver/`, e.g.:
    ```
    python make_jittable.py --data-config /hwwtaggervol/melissa-weaver/data/mq_ntuples/melissa_dataconfig_semilep_ttbarwjets.yaml -n networks/particle_net_pf_sv_4_layers_pyg_ef.py -m 05_10_ak8_ttbarwjets
    ```
    or
    ```
    python make_jittable.py --data-config models/particlenet_hww_inclv2_pre2/data/ak8_MD_vminclv2_pre2.yaml -n networks/particle_net_pf_sv_hybrid.py -m models/particlenet_hww_inclv2_pre2/data/net
    ```
  - Copy this jittable file, the config file with labels and the json file to the PR.
- Create a pod in the triton server and use `sudo` to pull changes from this repository.
  ```
  kubectl create -f tritonpod.yml
  ```
  and
  ```
  cd /triton/sonic-models/
  sudo git pull origin master
  ```

## Analysis

The output will be stored in ${ODIR}, e.g.: `/eos/uscms/store/user/cmantill/boostedhiggs/Nov4`.

### Luminosities

```
2017: nominal 41480.0
SingleElectron: 41476.02
SingleMuon: 41475.26
```

### Normalization

To convert to root files using:
```
python convert_to_root.py --dir ${ODIR} --ch ele,mu --odir rootfiles
```

### Histograms

The configs to make histograms are under `plot_configs`.

Make histograms with correct normalization:
```
python make_hists.py --year 2017 --odir ${TAG} --channels ele,mu --idir ${ODIR} --vars plot_configs/vars.yaml
```
and make stacked histograms with:
```
python plot_stacked_hists.py --year 2017 --odir ${TAG}
# e.g. for variable=cutflow with no data
python plot_stacked_hists.py --year 2017 --odir ${TAG} --var cutflow --nodata
```

You can also customize the `vars.yaml` file. For example:
```
python make_hists.py --year 2017 --odir ${CUSTOM_TAG} --channels ele,mu --idir ${ODIR} --vars plot_configs/genvars.yaml
```
and use `plot_1dhists.py` to create 1D hists for specific variables. Use `samples` to customize samples to compare.
```
python plot_1dhists.py --year 2017 --odir ${CUSTOM_TAG} --var gen_Hpt --samples GluGluHToWW_Pt-200ToInf_M-125,VH,VBFHToWWToLNuQQ_M-125_withDipoleRecoil --tag signal --logy
```

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
conda create -n coffea-env python=3.8

# activate the environment
conda activate coffea-env

# install packages
pip install numpy pandas scikit-learn coffea correctionlib pyarrow
pip install tritonclient['all']

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

