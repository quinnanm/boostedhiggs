"""
Methods for deriving input variables for the tagger.
Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan, Farouk Mokhtar
"""

from typing import Dict
from coffea.nanoevents.methods.base import NanoEventsArray
import awkward as ak
import numpy as np
import numpy.ma as ma

import json


def get_pfcands_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    fj_idx_lep,
    fatjet_label: str = "FatJetAK15",
    pfcands_label: str = "FatJetPFCands",
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extracts the pf_candidate features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    jet = ak.firsts(preselected_events[fatjet_label][fj_idx_lep])

    msk = preselected_events[pfcands_label].jetIdx == ak.firsts(fj_idx_lep)
    jet_ak_pfcands = preselected_events[pfcands_label][msk]
    jet_pfcands = (preselected_events.PFCands[jet_ak_pfcands.pFCandsIdx])

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.values_astype(jet_pfcands.eta > 0, int) * 2 - 1
    feature_dict["pfcand_etarel"] = eta_sign * (jet_pfcands.eta - jet.eta)
    feature_dict["pfcand_phirel"] = jet.delta_phi(jet_pfcands)
    feature_dict["pfcand_abseta"] = np.abs(jet_pfcands.eta)

    feature_dict["pfcand_pt_log_nopuppi"] = np.log(jet_pfcands.pt)
    feature_dict["pfcand_e_log_nopuppi"] = np.log(jet_pfcands.energy)

    pdgIds = jet_pfcands.pdgId
    feature_dict["pfcand_isEl"] = np.abs(pdgIds) == 11
    feature_dict["pfcand_isMu"] = np.abs(pdgIds) == 13
    feature_dict["pfcand_isChargedHad"] = np.abs(pdgIds) == 211
    feature_dict["pfcand_isGamma"] = np.abs(pdgIds) == 22
    feature_dict["pfcand_isNeutralHad"] = np.abs(pdgIds) == 130

    feature_dict["pfcand_charge"] = jet_pfcands.charge
    feature_dict["pfcand_VTX_ass"] = jet_pfcands.pvAssocQuality
    feature_dict["pfcand_lostInnerHits"] = jet_pfcands.lostInnerHits
    feature_dict["pfcand_quality"] = jet_pfcands.trkQuality

    feature_dict["pfcand_normchi2"] = np.floor(jet_pfcands.trkChi2)

    feature_dict["pfcand_dz"] = jet_pfcands.dz
    feature_dict["pfcand_dxy"] = jet_pfcands.d0
    feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
    feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err

    # btag vars
    for var in tagger_vars["pf_features"]["var_names"]:
        if "btag" in var:
            feature_dict[var] = jet_ak_pfcands[var[len("pfcand_"):]]

    # pfcand mask
    feature_dict["pfcand_mask"] = (~(ma.masked_invalid( ak.pad_none(feature_dict["pfcand_abseta"], tagger_vars["pf_points"]["var_length"], axis=1, clip=True).to_numpy() ).mask) ).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in tagger_vars["pf_features"]["var_names"]:
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["pf_points"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        if normalize:
            info = tagger_vars["pf_features"]["var_infos"][var]
            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    if normalize:
        var = "pfcand_normchi2"
        info = tagger_vars["pf_features"]["var_infos"][var]
        # finding what -1 transforms to
        chi2_min = -1 - info["median"] * info["norm_factor"]
        feature_dict[var][feature_dict[var] == chi2_min] = info["upper_bound"]
    return feature_dict


def get_svs_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    fj_idx_lep,
    fatjet_label: str = "FatJetAK15",
    svs_label: str = "JetSVsAK15",
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extracts the sv features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    jet = ak.firsts(preselected_events[fatjet_label][fj_idx_lep])
    msk = preselected_events[svs_label].jetIdx == ak.firsts(fj_idx_lep)
    jet_svs = preselected_events.SV[
        preselected_events[svs_label].sVIdx[
            (preselected_events[svs_label].sVIdx != -1)
            * (msk)
        ]
    ]

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.values_astype(jet_svs.eta > 0, int) * 2 - 1
    feature_dict["sv_etarel"] = eta_sign * (jet_svs.eta - jet.eta)
    feature_dict["sv_phirel"] = jet_svs.delta_phi(jet)
    feature_dict["sv_abseta"] = np.abs(jet_svs.eta)
    feature_dict["sv_mass"] = jet_svs.mass
    feature_dict["sv_pt_log"] = np.log(jet_svs.pt)

    feature_dict["sv_ntracks"] = jet_svs.ntracks
    feature_dict["sv_normchi2"] = jet_svs.chi2
    feature_dict["sv_dxy"] = jet_svs.dxy
    feature_dict["sv_dxysig"] = jet_svs.dxySig
    feature_dict["sv_d3d"] = jet_svs.dlen
    feature_dict["sv_d3dsig"] = jet_svs.dlenSig
    svpAngle = jet_svs.pAngle
    feature_dict["sv_costhetasvpv"] = -np.cos(svpAngle)

    feature_dict["sv_mask"] = (~(ak.pad_none(feature_dict["sv_etarel"], tagger_vars["sv_points"]["var_length"], axis=1, clip=True).to_numpy().mask)).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in tagger_vars["sv_features"]["var_names"]:
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["sv_points"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        if normalize:
            info = tagger_vars["sv_features"]["var_infos"][var]
            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict
