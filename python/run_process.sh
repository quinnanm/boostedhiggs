#!/bin/bash

"""
python process_histograms.py --sample "hww" --histogram "jet_kin"
python process_histograms.py --sample "hww" --histogram "met_kin"
python process_histograms.py --sample "hww" --histogram "cutflow"

python process_histograms.py --sample "tt" --histogram "jet_kin"
python process_histograms.py --sample "tt" --histogram "met_kin"
python process_histograms.py --sample "tt" --histogram "cutflow"

python process_histograms.py --sample "qcd" --histogram "jet_kin"
python process_histograms.py --sample "qcd" --histogram "met_kin"
python process_histograms.py --sample "qcd" --histogram "cutflow"

python process_histograms.py --sample "wjets" --histogram "jet_kin"
python process_histograms.py --sample "wjets" --histogram "met_kin"
python process_histograms.py --sample "wjets" --histogram "cutflow"
"""

python process_histograms.py --sample "electron" --histogram "jet_kin"
python process_histograms.py --sample "electron" --histogram "met_kin"
python process_histograms.py --sample "electron" --histogram "cutflow"

python process_histograms.py --sample "muon" --histogram "jet_kin"
python process_histograms.py --sample "muon" --histogram "met_kin"
python process_histograms.py --sample "muon" --histogram "cutflow"
