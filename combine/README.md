# Summary of the full pipeline

`python ../python/make_templates.py --year 2017 --channels mu --tag test`

`python create_datacard.py --years 2017 --channels mu --tag test`

`python produce_datacard.py --years 2017 --channels mu --tag test`

# Explanation below

## Step 1. Making combine templates

We use a python script to produce the templates ready for datacard creation. The script can be found under `python/` in the parent repository [here](https://github.com/cmantill/boostedhiggs/tree/main/python).

The output templates should be stored by default here under `templates/`.

## Step 2. Creating the datacard

The following line will create a rhalphalib Model from the hist templates stored under `templates/` (read above)
`python create_datacard.py --years 2017 --channels mu --tag test`

## Step 3. Producing the datacard

Run the following line in a cmsenv to build the datacard
`python produce_datacard.py --years 2017 --channels mu --tag test`
