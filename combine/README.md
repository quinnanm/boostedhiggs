# Making combine templates
We use the python script `make_templates.py` to produce the templates before datacard creation but first make sure to edit the `config_make_templates.yaml`.
```
python make_templates.py --years 2018,2017 --channels mu,ele --outdir templates/test
```
Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `hists_templates_{year}_{ch}.pkl`.

Then, The following line will create a rhalphalib Model from the hist templates stored under `templates/v1` and builds the datacard.
```
python create_datacard.py --years 2018,2017 --channels mu,ele --outdir templates/v1
```
Sample rhalphalib model produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/test/) as `model_2017_mu.pkl`.

# Making combine templates for asimov significance tests
To quickly run tests for Asimov significance for different regions in the phase space, we build the templates without caring about all systematics,
We use the python script `make_templates_sig.py` to produce the templates before datacard creation but first make sure to edit the `config_make_templates_sig.yaml`.
```
python make_templates_sig.py --years 2018,2017 --channels mu,ele --outdir templates/test
python create_datacard_wig.py --years 2018,2017 --channels mu,ele --outdir templates/v1
```
Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `hists_templates_{year}_{ch}.pkl`.

```
python create_datacard.py --years 2018,2017 --channels mu,ele --outdir templates/v1
```