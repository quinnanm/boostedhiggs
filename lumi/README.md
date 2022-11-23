# Computing luminosity from GoldenJson

1. Generate the lumi.csv using a GoldenJson.

For example, to generate the lumi2017.csv run the following command in a cmssw enviornment:
```
cd ../
brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml  -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json  -u /pb --byls --output-style csv -i Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt > lumi2017.csv 
```

2. Run the lumi processor to produce the output.pkl files.

3. Combine the output pickle files.

After editing the `dir` and `datasets` in the script, run it as:
```
python combine_lumi.py
```

4. Compute the luminosity.
After editing the `dir` in the script, run it as:
```
python compute_lumi.py
```
