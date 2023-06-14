# Making templates

We use the script here to produce the `.root` histograms (or templates).
```
python python/make_templates.py
```

### Running combine on the output

A sample datacard is found here `datacard_hww_sig_region.txt`.

We can run combine commands like,

```
combine -M AsymptoticLimits datacard_hww_sig_region.txt
```
