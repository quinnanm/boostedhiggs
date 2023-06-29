"""
Skimmer for ParticleNet tagger inputs.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""
import os
import pathlib
import warnings
from typing import Dict

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, dict_accumulator

from .get_tagger_inputs import get_lep_features, get_met_features

# from .run_tagger_inference import runInferenceTriton
from .utils import FILL_NONE_VALUE, add_selection_no_cutflow, tagger_gen_matching

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")

P4 = {
    "eta": "eta",
    "phi": "phi",
    "mass": "mass",
    "pt": "pt",
}


class InputProcessor(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, label, inference, output_location="./outfiles/"):
        """
        :param num_jets: Number of jets to save
        :type num_jets: int
        """

        """
        Skimming variables
        """
        self.label = label
        self.inference = inference
        self._output_location = output_location

        self.skim_vars = {
            "Event": {
                "event": "event",
            },
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            },
            "GenPart": [
                "fj_genjetmass",
                "fj_genRes_pt",
                "fj_genRes_eta",
                "fj_genRes_phi",
                "fj_genRes_mass",
                "fj_nprongs",
                "fj_ncquarks",
                "fj_lepinprongs",
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_leptauelvqq",
                "fj_H_VV_leptaumuvqq",
                "fj_H_VV_hadtauvqq",
                "fj_H_VV_unmatched",
                "fj_QCDb",
                "fj_QCDbb",
                "fj_QCDc",
                "fj_QCDcc",
                "fj_QCDothers",
                "fj_V_2q",
                "fj_V_elenu",
                "fj_V_munu",
                "fj_V_taunu",
                "fj_Top_bmerged",
                "fj_Top_2q",
                "fj_Top_elenu",
                "fj_Top_munu",
                "fj_Top_hadtauvqq",
                "fj_Top_leptauelvnu",
                "fj_Top_leptaumuvnu",
            ],
            # formatted to match weaver's preprocess.json
            "MET": {
                "met_features": {
                    "var_names": [
                        "met_relpt",
                        "met_relphi",
                    ],
                },
                "met_points": {"var_length": 1},
            },
            "Lep": {
                "fj_features": {
                    "fj_lep_dR",
                    "fj_lep_pt",
                    "fj_lep_iso",
                    "fj_lep_miniiso",
                },
            },
        }

        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

        self.fatjet_label = "FatJet"
        self.pfcands_label = "FatJetPFCands"
        self.svs_label = "FatJetSVs"

        self._accumulator = dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            PATH = f"{self._output_location}/parquet/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, f"{PATH}/{fname}.parquet")

    def dump_root(self, jet_vars: Dict[str, np.array], fname: str) -> None:
        """
        Saves ``jet_vars`` dict as a rootfile to './outroot'
        """
        local_dir = os.path.abspath(os.path.join(".", "outroot"))
        os.system(f"mkdir -p {local_dir}")

        with uproot.recreate(f"{local_dir}/{fname}", compression=uproot.LZ4(4)) as rfile:
            rfile["Events"] = ak.Array(jet_vars)
            # rfile["Events"].show()

    def to_pandas_lists(self, events: Dict[str, np.array]) -> pd.DataFrame:
        """
        Convert our dictionary of numpy arrays into a pandas data frame.
        Uses lists for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        output = pd.DataFrame()
        for field in ak.fields(events):
            if "sv_" in field or "pfcand_" in field:
                output[field] = events[field].tolist()
            else:
                output[field] = ak.to_numpy(ak.flatten(events[field], axis=None))

        return output

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def process(self, events: ak.Array):
        import time

        start = time.time()

        def build_p4(cand):
            return ak.zip(
                {
                    "pt": cand.pt,
                    "eta": cand.eta,
                    "phi": cand.phi,
                    "mass": cand.mass,
                    "charge": cand.charge,
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )

        electrons = events["Electron"][events["Electron"].pt > 40]
        muons = events["Muon"][events["Muon"].pt > 30]
        leptons = ak.concatenate([electrons, muons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
        fatjets = events[self.fatjet_label]
        candidatelep_p4 = build_p4(ak.firsts(leptons))

        fj_idx_lep = ak.argmin(fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        fatjet = ak.firsts(fatjets[fj_idx_lep])

        # selection
        selection = PackedSelection()
        add_selection_no_cutflow("fjselection", (fatjet.pt > 200), selection)

        if np.sum(selection.all(*selection.names)) == 0:
            return {}

        # variables
        FatJetVars = {
            f"fj_{key}": ak.fill_none(fatjet[var], FILL_NONE_VALUE) for (var, key) in self.skim_vars["FatJet"].items()
        }
        LepVars = {
            **get_lep_features(
                self.skim_vars["Lep"],
                events,
                fatjet,
                candidatelep_p4,
            ),
        }

        METVars = {
            **get_met_features(
                self.skim_vars["MET"],
                events,
                fatjet,
                "MET",
                normalize=False,
            ),
        }

        genparts = events.GenPart
        matched_mask, genVars = tagger_gen_matching(
            events,
            genparts,
            fatjet,
            # candidatelep_p4,
            self.skim_vars["GenPart"],
            label=self.label,
        )
        add_selection_no_cutflow("gen_match", matched_mask, selection)

        skimmed_vars = {**FatJetVars, **genVars, **METVars, **LepVars}

        # apply selections
        skimmed_vars = {
            key: np.squeeze(np.array(value[selection.all(*selection.names)])) for (key, value) in skimmed_vars.items()
        }

        # Farouk fix this
        """
        pnet_vars = runInferenceTriton(
            self.tagger_resources_path,
            events[selection.all(*selection.names)],
            ak15=False,
        )

        pnet_vars_jet = {**{key: value[:, jet_idx] for (key, value) in pnet_vars.items()}}
        """
        # fill inference
        if self.inference:
            from .run_tagger_inference import runInferenceTriton

            for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap"]:
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path,
                    events[selection.all(*selection.names)],
                    fj_idx_lep[selection.all(*selection.names)],
                    model_name=model_name,
                )

                skimmed_vars = {
                    **skimmed_vars,
                    **{key: value for (key, value) in pnet_vars.items()},
                }

        # convert output to pandas
        df = self.ak_to_pandas(skimmed_vars)

        print(f"convert: {time.time() - start:.1f}s")

        print(df)

        # save the output
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        self.save_dfs_parquet(fname, df)

        print(f"dumped: {time.time() - start:.1f}s")

        return {}

    def postprocess(self, accumulator):
        pass
