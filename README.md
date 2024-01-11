# Boosted Higgs To WW

- [Boosted Higgs To WW](#boosted-higgs-to-ww)
  * [Repository](#repository)
    + [Data fileset](#data-fileset)
  * [Submitting condor jobs](#submitting-condor-jobs)
    + [First time setup](#first-time-setup)
    + [Submitting jobs](#submitting-jobs)
      - [Testing jobs locally per single sample](#testing-jobs-locally-per-single-sample-)
      - [Testing jobs with inference (and triton server running)](#testing-jobs-with-inference--and-triton-server-running--)
      - [Testing jobs locally over multiple samples specified in the config](#testing-jobs-locally-over-multiple-samples-specified-in-the-config-)
  * [Triton server setup](#triton-server-setup)
    + [Running the server](#running-the-server)
    + [First time setup with a new model](#first-time-setup-with-a-new-model)
  * [Analysis](#analysis)

## Repository

We use pre-commit:
```
# install pre-commit
pip install pre-commit

# setup pre-commit hooks
pre-commit install
```

Before pushing changes to git make sure to run:
```
pre-commit run -a
```

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
python condor/submit.py --year 2017 --tag ${TAG} --config samples_inclusive.yaml --key mc --pfnano v2_2 --channels mu,ele --submit
```
where:
- year: this determines which fileset to read
- tag: is a tag to the jobs (usually a date or something more descriptive)
- config: a yaml file that contains the names of the samples to run and the number of files per job for that sample
- pfnano: pfnano version
- number of files per job: if given all of the samples will use these number of files per job
- script that runs processor: is `run.py` by default
--no-inference: do not use inference
--inference: (true by default)

e.g.
```
python3 condor/submit.py --year 2017 --tag ${TAG} --config samples_inclusive.json --key mc --slist GluGluHToWW_Pt-200ToInf_M-125,TTToSemiLeptonic --submit --no-inference
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
python run.py --year 2017 --processor hww --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125 --local --channel=ele
```

#### Testing jobs with inference (and triton server running):
```
python run.py --year 2017 --processor hww --n 1 --starti 0 --sample GluGluHToWWToLNuQQ --local --inference
```

#### Testing jobs locally over multiple samples:
```
python run.py --year 2017 --processor hww --n 1 --starti 0 --sample GluGluHToWW_Pt-200ToInf_M-125,GluGluHToWWToLNuQQ
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
  kubectl exec -it tritonpodi -- /bin/bash
  cd /triton/sonic-models/
  sudo git pull origin master
  kubectl delete pod tritonpodi
  ```

## Analysis

The output will be stored in ${ODIR}, e.g.: `/eos/uscms/store/user/cmantill/boostedhiggs/Nov4`.

### Luminosities

```
2016:
nominal: 16830.0
SingleElectron: 16809.97
SingleMuon: 16810.81

2016APV:
nominal: 19500.0
SingleElectron: 19492.72
SingleMuon: 19436.16

2017:
nominal 41480.0
SingleElectron: 41476.02
SingleMuon: 41475.26

2018:
nominal: 59830.0
EGamma: 59816.23
SingleMuon: 59781.96

Run2:
nominal: 137640.0
ele: 137594.94
mu: 137504.19
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


# Weaver ParT-Finetuning

To produce the ntuples, use the `inputprocessor.py` by runing the commands in `run_skimmer.sh` (you may also use `condor/tagger_submit.py` to submit jobs to produce the ntuples faster).

For the weaver setup:
```
conda create -n weaver python=3.8

conda activate weaver

pip install torch==1.10
pip install numba
pip install weaver-core
pip install tensorboard
pip install setuptools==59.5.0
```
