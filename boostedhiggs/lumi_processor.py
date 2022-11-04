from collections import defaultdict
import pickle as pkl
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import shutil
import pathlib
from typing import List, Optional
import pyarrow.parquet as pq

import importlib.resources

import coffea
from coffea import processor, lumi_tools
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

from boostedhiggs.utils import match_HWW, getParticles, match_V, match_Top
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    add_VJets_kFactors,
    add_jetTriggerSF,
    add_lepton_weight,
    add_pileup_weight,
)
from boostedhiggs.btag import btagWPs, BTagCorrector

from .run_tagger_inference import runInferenceTriton

import warnings

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class LumiProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        output_location="./outfiles/",
    ):

        self._year = year
        self._yearmod = yearmod

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata["dataset"]
        isMC = hasattr(events, "genWeight")
        nevents = len(events)
        
        # lumilist = coffea.lumi_tools.LumiList(events.run.to_numpy(), events.luminosityBlock.to_numpy())
        lumilist = set(zip(events.run, events.luminosityBlock))
        print(lumilist)

        # return dictionary with cutflows
        return {dataset: {"mc": isMC, self._year + self.year_mod: {"lumilist": lumilist}}}

    def postprocess(self, accumulator):
        return accumulator
