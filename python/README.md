# Making stacked histograms

We build stacked histograms after processing the parquets into an `events_dict` object.

To make stacked histograms run the script `make_stacked_hists.py` but make sure to first edit the `config_make_stacked_hists.yaml`.
```
python make_stacked_hists.py --years 2016APV,2017 --channels ele,mu --samples_dir Apr12_presel_ --outpath test/
```

Notes
- To build the `events_dict` object without making plots, then only pass `--make_events_dict`
- If you had already built the `events_dict` object and you just want to make plots, then only pass `--plot_hists`
- To start from scratch add both args e.g.
```
python make_hists.py --years 2017 --channels ele,mu --make_events_dict --plot_hists
```

To make 1d-histograms from the `events_dict` see e.g. binder/July6_regressed_mass.ipynb

To make 2d-histograms from the `events_dict` see e.g. binder/July6_VBF_exploration.ipynb
