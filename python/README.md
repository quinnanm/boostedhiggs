# Making stacked histograms

We build histograms after processing the parquets into an `events_dict` object.

To make stacked histograms use the script `make_hists.py` but make sure to first edit the `make_hists_config.yaml` e.g.
```
python make_hists.py --years 2016APV,2017 --channels ele,mu --samples_dir April_presel_ --outpath test/
```

Notes
- To build the `events_dict` object without making plots, then only pass `--make_events_dict`
- If you had already built the `events_dict` object and you just want to make plots, then only pass `--plot_hists`
- To do both you can run e.g.
```
python make_hists.py --years 2017 --channels ele,mu --make_events_dict --plot_hists
```

To make 1d-histograms from the `events_dict` see e.g. binder/July6_regressed_mass.ipynb

To make 2d-histograms from the `events_dict` see e.g. binder/July6_VBF_exploration.ipynb

# Making combine templates

We use a python script to produce the `.root` histograms (or templates).
```
python make_templates.py
```
