import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import shutil
import pathlib
from typing import List, Optional
import pyarrow.parquet as pq

from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret


class HwwProcessor(processor.ProcessorABC):

    def __init__(self, year="2017", yearmod="", channels=["ele", "mu", "had"], output_location="./"):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._output_location = output_location

        # define variables to save for each channel
        self._skimvars = {
            'ele': [
                "lepton_pt"
            ],
            'mu': [
                "lepton_pt",
            ],
            'had': [
                "fatjet_pt",
            ],
        }

        # trigger paths
        self._HLTs = {
            2016: {
                'ele': [
                    "Ele27_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon175",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "TkMu50",
                    "IsoMu24",
                    "IsoTkMu24",
                    # "Mu55",
                    # "Mu15_IsoVVVL_PFHT600" # VVL
                ],
                'had': [
                    "PFHT800",
                    "PFHT900",
                    "AK8PFJet360_TrimMass30",
                    "AK8PFHT700_TrimR0p1PT0p03Mass50",
                    "PFHT650_WideJetMJJ950DEtaJJ1p5",
                    "PFHT650_WideJetMJJ900DEtaJJ1p5",
                    "PFJet450",
                ],
            },
            2017: {
                'ele': [
                    "Ele35_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon200",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "IsoMu27",
                    "OldMu100",
                    "TkMu100",
                    # "Mu15_IsoVVVL_PFHT600", # VVL
                ],
                'had': [
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                ],
            },
            2018: {
                'ele': [
                    "Ele32_WPTight_Gsf",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Photon200",
                    # "Ele50_CaloIdVT_GsfTrkIdT_PFJet165", # extra
                    # "Ele15_IsoVVVL_PFHT600", # VVL
                ],
                'mu': [
                    "Mu50",
                    "IsoMu24",
                    "OldMu100",
                    "TkMu100",
                    # "Mu15_IsoVVVL_PFHT600", # VVL
                ],
                'had': [
                    "PFHT1050",
                    "AK8PFJet400_TrimMass30",
                    "AK8PFJet420_TrimMass30",
                    "AK8PFHT800_TrimMass50",
                    "PFJet500",
                    "AK8PFJet500",
                ],
            }
        }[int(self._year)]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        self._metfilters = {
            2016: [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                "eeBadScFilter",
            ],
            2017: [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                # "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
            2018:  [
                "goodVertices",
                "globalSuperTightHalo2016Filter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter",
                # "BadChargedCandidateFilter",
                "eeBadScFilter",
                "ecalBadCalibFilter",
            ],
        }[int(self._year)]

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self._btagWPs = {
            '2016preVFP': {
                'loose': 0.0508,
                'medium': 0.2598,
                'tight': 0.6502,
            },
            '2016postVFP': {
                'loose': 0.0480,
                'medium': 0.2489,
                'tight': 0.6377,
            },
            '2017': {
                'loose': 0.0532,
                'medium': 0.3040,
                'tight': 0.7476,
            },
            '2018': {
                'loose': 0.0490,
                'medium': 0.2783,
                'tight': 0.7100,
            },
        }[year + yearmod]

        self.selections = {}
        self.cutflows = {}

    @property
    def accumulator(self):
        return self._accumulator

    def h5store(self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs

    def save_dfs(self, fname, dfs_dict, isMC, dataset, sumgenweight, cutflows):
        subdirs = []
        store = pd.HDFStore(fname)
        if self._output_location is not None:
            for gname, out in dfs_dict.items():
                if isMC:
                    metadata = dict(sumgenweight=sumgenweight, year=self._year, mc=isMC, dataset=dataset, cutflow=cutflows[gname])
                else:
                    metadata = dict(year=self._year, mc=isMC, dataset=dataset, cutflows=cutflows)
                store_fin = self.h5store(store, out, fname, gname, **metadata)
            store.close()
            self.dump_table(fname, self._output_location, subdirs)
        else:
            print("self._output_location is None")
            store.close()

    def dump_table(self, fname: str, location: str, subdirs: Optional[List[str]] = None) -> None:
        """
        Saves locally first, then:
        if ``location`` starts with "root://", will xrdcp to ``location``/``[subdirs]``/.
        else will copy normally to ``location``/``[subdirs]``/.
        """

        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                )
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1:],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            # TODO: need to fix this
            if not os.path.samefile(local_file, destination):
                shutil.copy2(local_file, destination)
            else:
                fname = "condor_" + fname
                destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                shutil.copy2(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = None):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = [channel] if channel else self._channels
        for ch in self._channels:
            self.selections[ch].add(name, sel)
            self.cutflows[ch][name] = np.sum(self.selections[ch].all(*self.selections[ch].names))

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata['dataset']
        isMC = hasattr(events, "genWeight")
        sumgenweight = ak.sum(events.genWeight) if isMC else 0
        nevents = len(events)

        # empty selections and cutflows
        self.selections = {}
        self.cutflows = {}
        for ch in self._channels:
            self.selections[ch] = PackedSelection()
            self.cutflows[ch] = {}
            self.cutflows[ch]["all"] = nevents

        # trigger
        triggers = {}
        for ch in self._channels:
            if ch == "had" and isMC:
                trigger = np.ones(nevents, dtype='bool')
            else:
                # apply trigger to both data and MC (except for hadronic channel)
                trigger = np.zeros(len(events), dtype='bool')
                for t in self._HLTs[ch]:
                    if t in events.HLT.fields:
                        trigger = trigger | events.HLT[t]
            self.add_selection("trigger", trigger, ch)
            del trigger

        # metfilters
        metfilters = np.ones(nevents, dtype='bool')
        for mf in self._metfilters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        self.add_selection("metfilters", metfilters)

        # muons
        goodmuon = (
            (events.Muon.pt > 25)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.5)
            & (np.abs(events.Muon.dxy) < 0.2)
            & events.Muon.mediumId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        lowptmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.5)
            & (np.abs(events.Muon.dxy) < 0.2)
            & events.Muon.looseId
        )
        nlowptmuons = ak.sum(lowptmuon, axis=1)

        # electrons
        goodelectron = (
            (events.Electron.pt > 25)
            & (abs(events.Electron.eta) < 2.5)
            & ((1.44 < np.abs(events.Electron.eta)) | (np.abs(events.Electron.eta) > 1.57))
            & (events.Electron.mvaFall17V2noIso_WP90)
        )
        nelectrons = ak.sum(goodelectron, axis=1)
        lowptelectron = (
            (events.Electron.pt > 10)
            & ((1.44 < np.abs(events.Electron.eta)) | (np.abs(events.Electron.eta) > 1.57))
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nlowptelectrons = ak.sum(lowptelectron, axis=1)

        goodleptons = ak.concatenate([events.Muon[goodmuon], events.Electron[goodelectron]], axis=1)
        candidatelep = ak.firsts(goodleptons[ak.argsort(goodleptons.pt)])
        candidatelep_p4 = ak.zip(
            {
                "pt": candidatelep.pt,
                "eta": candidatelep.eta,
                "phi": candidatelep.phi,
                "mass": candidatelep.mass,
                "charge": candidatelep.charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            out = {}
            for var in self._skimvars[ch]:
                if var == "lepton_pt":
                    value = pad_val(candidatelep.pt, 0)
                else:
                    continue
                out[var] = value

            # print arrays and selections to debug
            # print(out)
            # print(selections[ch].all(*selections[ch].names))

            # apply selections
            output[ch] = {
                key: value[self.selections[ch].all(*self.selections[ch].names)] for (key, value) in out.items()
            }

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + "-out.hdf5"
        self.save_dfs(fname, output, isMC, dataset, sumgenweight, self.cutflows)

        return {}

    def postprocess(self, accumulator):
        return accumulator
