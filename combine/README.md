# Summary of the full pipeline

## Step 1. Making combine templates
We use the python script `make_templates.py` to produce the templates before datacard creation but first make sure to edit the `config_make_templates.yaml`.
```
python make_templates.py --years 2018,2017 --channels mu,ele --outdir templates/test
```
Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/test/) as `hists_templates_{year}_{ch}.pkl`.

## Step 2. Creating the datacard

The following line will create a rhalphalib Model from the hist templates stored under `templates/`.
```
python create_datacard.py --years 2018,2017 --channels mu,ele --outdir templates/test
```
Sample rhalphalib model produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/test/) as `model_2017_mu.pkl`.

## Step 3. Producing the datacard

Run the following line in a cmsenv to build the datacard.
```
python produce_datacard.py --years 2018,2017 --channels mu,ele --outdir templates/test
```
Sample datacards produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/test/datacards).
