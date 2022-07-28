"""
Methods for running the tagger inference.
Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan, Farouk Mokhtar
"""

from typing import Dict

import numpy as np
from scipy.special import softmax
import awkward as ak
from coffea.nanoevents.methods.base import NanoEventsArray

import json

# import onnxruntime as ort

import time

import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http

from tqdm import tqdm

from .get_tagger_inputs import get_pfcands_features, get_svs_features


# adapted from https://github.com/lgray/hgg-coffea/blob/triton-bdts/src/hgg_coffea/tools/chained_quantile.py
class wrapped_triton:
    def __init__(
        self,
        model_url: str,
        batch_size: int,
        torchscript: bool = True,
    ) -> None:
        fullprotocol, location = model_url.split("://")
        _, protocol = fullprotocol.split("+")
        address, model, version = location.split("/")

        self._protocol = protocol
        self._address = address
        self._model = model
        self._version = version

        self._batch_size = batch_size
        self._torchscript = torchscript

    def __call__(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if self._protocol == "grpc":
            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False)
            triton_protocol = triton_grpc
        elif self._protocol == "http":
            client = triton_http.InferenceServerClient(
                url=self._address,
                verbose=False,
                concurrency=12,
            )
            triton_protocol = triton_http
        else:
            raise ValueError(f"{self._protocol} does not encode a valid protocol (grpc or http)")

        # manually split into batches for gpu inference
        input_size = input_dict[list(input_dict.keys())[0]].shape[0]
        print(f"size of input = {input_size}")

        outs = [
            self._do_inference(
                {key: input_dict[key][batch: batch + self._batch_size] for key in input_dict},
                triton_protocol,
                client,
            )
            for batch in tqdm(
                range(0, input_dict[list(input_dict.keys())[0]].shape[0], self._batch_size)
            )
        ]

        return np.concatenate(outs) if input_size > 0 else outs

    def _do_inference(
        self, input_dict: Dict[str, np.ndarray], triton_protocol, client
    ) -> np.ndarray:
        # Infer
        inputs = []

        for key in input_dict:
            input = triton_protocol.InferInput(key, input_dict[key].shape, "FP32")
            input.set_data_from_numpy(input_dict[key])
            inputs.append(input)

        out_name = "softmax__0" if self._torchscript else "softmax"

        output = triton_protocol.InferRequestedOutput(out_name)

        request = client.infer(
            self._model,
            model_version=self._version,
            inputs=inputs,
            outputs=[output],
        )

        return request.as_numpy(out_name)


def runInferenceTriton(
    tagger_resources_path: str, events: NanoEventsArray, fj_idx_lep
) -> dict:
    total_start = time.time()

    with open(f"{tagger_resources_path}/triton_config.json") as f:
        triton_config = json.load(f)

    with open(f"{tagger_resources_path}/{triton_config['model_name']}.json") as f:
        tagger_vars = json.load(f)

    triton_model = wrapped_triton(
        triton_config["model_url"], triton_config["batch_size"], torchscript=True
    )

    fatjet_label = "FatJet"
    pfcands_label = "FatJetPFCands"
    svs_label = "FatJetSVs"
    jet_label = "ak8"

    # prepare inputs for both fat jets
    tagger_inputs = []

    print('getting cands')
    feature_dict = {
        **get_pfcands_features(tagger_vars, events, fj_idx_lep, fatjet_label, pfcands_label),
        **get_svs_features(tagger_vars, events, fj_idx_lep, fatjet_label, svs_label),
    }

    print('got cands')

    for input_name in tagger_vars["input_names"]:
        for key in tagger_vars[input_name]["var_names"]:
            np.expand_dims(feature_dict[key], 1)

    print('building inputs')

    tagger_inputs = {
        f"{input_name}__{i}": np.concatenate(
            [
                np.expand_dims(feature_dict[key], 1)
                for key in tagger_vars[input_name]["var_names"]
            ],
            axis=1,
        )
        for i, input_name in enumerate(tagger_vars["input_names"])
    }

    # run inference for both fat jets
    tagger_outputs = []
    print(f"Running inference for candidate Jet")
    start = time.time()
    tagger_outputs = triton_model(tagger_inputs)
    time_taken = time.time() - start
    print(f"Inference took {time_taken:.1f}s")
    return tagger_outputs

    pnet_vars_list = []
    if len(tagger_outputs):
        pnet_vars_list.append(
            {
                f"{jet_label}FatJetParticleNetHWWMD_probQCD": np.sum(
                    tagger_outputs[:, :5], axis=1
                ),
                f"{jet_label}FatJetParticleNetHWWMD_probHWW3q": tagger_outputs[:, -2],
                f"{jet_label}FatJetParticleNetHWWMD_probHWW4q": tagger_outputs[:, -1],
                f"{jet_label}FatJetParticleNetHWWMD_THWW4q": (
                    tagger_outputs[:, -2] + tagger_outputs[:, -1]
                )
                / np.sum(tagger_outputs[jet_idx], axis=1),
            }
        )
    else:
        pnet_vars_list.append(
            {
                f"{jet_label}FatJetParticleNetHWWMD_probQCD": np.array([]),
                f"{jet_label}FatJetParticleNetHWWMD_probHWW3q": np.array([]),
                f"{jet_label}FatJetParticleNetHWWMD_probHWW4q": np.array([]),
                f"{jet_label}FatJetParticleNetHWWMD_THWW4q": np.array([]),
            }
        )

    pnet_vars_combined = {
        key: np.concatenate(
            [pnet_vars_list[0][key][:, np.newaxis], pnet_vars_list[1][key][:, np.newaxis]], axis=1
        )
        for key in pnet_vars_list[0]
    }

    print(f"Total time taken: {time.time() - total_start:.1f}s")
    print('pnet_vars_combined', pnet_vars_combined.items())
    return pnet_vars_combined
    # return tagger_outputs
