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
from .utils import FILL_NONE_VALUE, add_selection_no_cutflow, bkgs, others, sigs, tagger_gen_matching

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
                "fj_nquarks",
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_leptauelvqq",
                "fj_H_VV_leptaumuvqq",
                "fj_H_VV_hadtauvqq",
                "fj_QCDb",
                "fj_QCDbb",
                "fj_QCDc",
                "fj_QCDcc",
                "fj_QCDothers",
                "fj_V_2q",
                "fj_V_elenu",
                "fj_V_munu",
                "fj_V_taunu",
                "fj_Top_nquarksnob",
                "fj_Top_nbquarks",
                "fj_Top_ncquarks",
                "fj_Top_nleptons",
                "fj_Top_nele",
                "fj_Top_nmu",
                "fj_Top_ntau",
                "fj_Top_taudecay",
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

    def save_dfs_parquet(self, df, fname):
        if self._output_location is not None:
            PATH = f"{self._output_location}/parquet/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            table = pa.Table.from_pandas(df)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, f"{PATH}/{fname}.parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def dump_root(self, skimmed_vars: Dict[str, np.array], fname: str) -> None:
        """
        Saves ``jet_vars`` dict as a rootfile to './outroot'
        """
        local_dir = os.path.abspath(os.path.join(self._output_location, "outroot"))
        os.system(f"mkdir -p {local_dir}")

        with uproot.recreate(f"{local_dir}/{fname}.root", compression=uproot.LZ4(4)) as rfile:
            rfile["Events"] = ak.Array(skimmed_vars)
            rfile["Events"].show()

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
        # add_selection_no_cutflow("gen_match", matched_mask, selection)

        skimmed_vars = {**FatJetVars, **{"matched_mask": matched_mask}, **genVars, **METVars, **LepVars}

        # apply selections
        skimmed_vars = {
            key: np.squeeze(np.array(value[selection.all(*selection.names)])) for (key, value) in skimmed_vars.items()
        }

        # fill inference
        if self.inference:
            from .run_tagger_inference import runInferenceTriton

            for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path,
                    events[selection.all(*selection.names)],
                    fj_idx_lep[selection.all(*selection.names)],
                    model_name=model_name,
                )

                # pnet_df = self.ak_to_pandas(pnet_vars)
                pnet_df = pd.DataFrame(pnet_vars)

                num = pnet_df[sigs].sum(axis=1)
                den = pnet_df[sigs].sum(axis=1) + pnet_df[bkgs + others].sum(axis=1)
                print("den1", den)

                den = pnet_df.sum(axis=1)
                print("den2", den)

                scores = {"fj_ParT_inclusive_score": (num / den).values}
                reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}

                hidNeurons = {}
                for key in pnet_vars:
                    if "hidNeuron" in key:
                        hidNeurons[key] = pnet_vars[key]

                skimmed_vars = {**skimmed_vars, **scores, **reg_mass, **hidNeurons}

        for key in skimmed_vars:
            skimmed_vars[key] = skimmed_vars[key].squeeze()

        # convert output to pandas
        df = pd.DataFrame(skimmed_vars)

        df = df.dropna()  # very few events would have genjetmass NaN for some reason

        print(f"convert: {time.time() - start:.1f}s")

        print(df)

        # save the output
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        self.save_dfs_parquet(df, fname)

        print(f"dump parquet: {time.time() - start:.1f}s")

        # TODO: drop NaNs from rootfiles
        self.dump_root(skimmed_vars, fname)
        print(f"dump rootfile: {time.time() - start:.1f}s")

        # for now do something like this to dump the parquets in root
        # OUTPATH = "../datafiles/ntuples/"

        # for sample in samples:
        #     print(sample)

        #     for file in os.listdir(f"{OUTPATH}/{sample}/train/"):
        #         if "parquet" not in file:
        #             continue

        #         d = pd.read_parquet(f"{OUTPATH}/{sample}/train/{file}")

        #         with uproot.recreate(f"{OUTPATH}/{sample}/train/out.root", compression=uproot.LZ4(4)) as rfile:
        #             rfile["Events"] = ak.Array(d.to_dict(orient="list", index=True))
        #             rfile["Events"].show()

        #     for file in os.listdir(f"{OUTPATH}/{sample}/test/"):
        #         if "parquet" not in file:
        #             continue

        #         d = pd.read_parquet(f"{OUTPATH}/{sample}/test/{file}")

        #         with uproot.recreate(f"{OUTPATH}/{sample}/test/out.root", compression=uproot.LZ4(4)) as rfile:
        #             rfile["Events"] = ak.Array(d.to_dict(orient="list", index=True))
        #             rfile["Events"].show()
        #     print("--------------------------")

        return {}

    def postprocess(self, accumulator):
        pass
