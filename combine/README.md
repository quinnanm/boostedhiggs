# Making combine templates
We use the python script `make_templates.py` to produce the templates before datacard creation but first make sure to edit the `config_make_templates.yaml`.
```
python make_templates.py --years 2018,2017,2016APV,2016 --channels mu,ele --outdir templates/v1
```
Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `hists_templates_{year}_{ch}.pkl`.

Then, The following line will create a rhalphalib Model from the hist templates stored under `templates/v1` and builds the datacard.
```
python create_datacard.py --years 2018,2017,2016APV,2016 --channels mu,ele --outdir templates/v1
```
Sample rhalphalib model produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `model_2017_mu.pkl`.
