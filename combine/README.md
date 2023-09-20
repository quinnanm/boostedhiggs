# Summary of the full pipeline

```
python make_templates.py --year 2017 --channels mu --tag test

python create_datacard.py --years 2017 --channels mu --tag test

python produce_datacard.py --years 2017 --channels mu --tag test
```

# Explanation below

## Step 1. Making combine templates

We use the python script `make_templates.py` to produce the templates ready for datacard creation. The output templates should be stored by default here under `templates/`.

Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `hists_templates_2017_mu.pkl`.

## Step 2. Creating the datacard

The following line will create a rhalphalib Model from the hist templates stored under `templates/` (read above)
`python create_datacard.py --years 2017 --channels mu --tag test`

Sample rhalphalib model produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/) as `model_2017_mu.pkl`.

## Step 3. Producing the datacard

Run the following line in a cmsenv to build the datacard
`python produce_datacard.py --years 2017 --channels mu --tag test`

Sample datacards produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v1/datacards).
