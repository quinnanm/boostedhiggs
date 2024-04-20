from typing import Union

import awkward as ak
import numpy as np
from coffea.nanoevents.methods import candidate, vector
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

ak.behavior.update(vector.behavior)


d_PDGID = 1
c_PDGID = 4
b_PDGID = 5
g_PDGID = 21
TOP_PDGID = 6

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

GAMMA_PDGID = 22
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

PI_PDGID = 211
PO_PDGID = 221
PP_PDGID = 111

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]

FILL_NONE_VALUE = -99999

JET_DR = 0.8


def to_label(array: ak.Array) -> ak.Array:
    return ak.values_astype(array, np.int32)


def get_pid_mask(
    genparts: GenParticleArray,
    pdgids: Union[int, list],
    ax: int = 2,
    byall: bool = True,
) -> ak.Array:
    """
    Get selection mask for gen particles matching any of the pdgIds in ``pdgids``.
    If ``byall``, checks all particles along axis ``ax`` match.
    """
    gen_pdgids = abs(genparts.pdgId)

    if type(pdgids) is list:
        mask = gen_pdgids == pdgids[0]
        for pdgid in pdgids[1:]:
            mask = mask | (gen_pdgids == pdgid)
    else:
        mask = gen_pdgids == pdgids

    return ak.all(mask, axis=ax) if byall else mask


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


# pad array with given value
def pad_val(
    arr: ak.Array,
    target: int,
    value: float,  # value can also be Bool variable
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    padded_arr = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    # pad_none will fill the array to target length with "None" for dedicated axis
    # "clip" means cut the array to the target length or not
    # fill_none will replace "None" value to some value
    return padded_arr.to_numpy() if to_numpy else padded_arr


def match_H(genparts: GenParticleArray, fatjet: FatJetArray):
    """Gen matching for Higgs samples"""
    higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # only select events that match an specific decay
    # matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)][:, 0]
    matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)]
    matched_higgs_mask = ak.any(fatjet.delta_r(matched_higgs) < 0.8, axis=1)

    matched_higgs = ak.firsts(matched_higgs)

    matched_higgs_children = matched_higgs.children
    higgs_children = higgs.children

    children_mask = get_pid_mask(matched_higgs_children, [W_PDGID], byall=False)
    is_hww = ak.any(children_mask, axis=1)

    # order by mass, select lower mass child as V* and higher as V
    matched_higgs_children = matched_higgs_children[children_mask]
    children_mass = matched_higgs_children.mass
    v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=1, keepdims=True)])
    v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=1, keepdims=True)])

    # VV daughters
    # requires coffea-0.7.21
    all_daus = higgs_children.distinctChildrenDeep
    all_daus = ak.flatten(all_daus, axis=2)
    all_daus_flat = ak.flatten(all_daus, axis=2)
    all_daus_flat_pdgId = abs(all_daus_flat.pdgId)

    # the following tells you about the decay
    num_quarks = ak.sum(all_daus_flat_pdgId <= b_PDGID, axis=1)
    num_leptons = ak.sum(
        (all_daus_flat_pdgId == ELE_PDGID) | (all_daus_flat_pdgId == MU_PDGID) | (all_daus_flat_pdgId == TAU_PDGID),
        axis=1,
    )
    num_electrons = ak.sum(all_daus_flat_pdgId == ELE_PDGID, axis=1)
    num_muons = ak.sum(all_daus_flat_pdgId == MU_PDGID, axis=1)
    num_taus = ak.sum(all_daus_flat_pdgId == TAU_PDGID, axis=1)

    # the following tells you about the matching
    # prongs except neutrino
    neutrinos = (
        (all_daus_flat_pdgId == vELE_PDGID) | (all_daus_flat_pdgId == vMU_PDGID) | (all_daus_flat_pdgId == vTAU_PDGID)
    )

    leptons = (all_daus_flat_pdgId == ELE_PDGID) | (all_daus_flat_pdgId == MU_PDGID) | (all_daus_flat_pdgId == TAU_PDGID)

    # num_m: number of matched leptons
    # number of quarks excludes neutrino and leptons
    num_m_quarks = ak.sum(fatjet.delta_r(all_daus_flat[~neutrinos & ~leptons]) < JET_DR, axis=1)
    num_m_leptons = ak.sum(fatjet.delta_r(all_daus_flat[leptons]) < JET_DR, axis=1)
    num_m_cquarks = ak.sum(fatjet.delta_r(all_daus_flat[all_daus_flat.pdgId == b_PDGID]) < JET_DR, axis=1)

    lep_daughters = all_daus_flat[leptons]
    # parent = ak.firsts(lep_daughters[fatjet.delta_r(lep_daughters) < JET_DR].distinctParent)
    parent = ak.firsts(lep_daughters.distinctParent)
    iswlepton = parent.mass == v.mass
    iswstarlepton = parent.mass == v_star.mass

    genVars = {"fj_genH_pt": ak.fill_none(higgs.pt, FILL_NONE_VALUE)}

    genVVars = {
        "fj_genH_jet": fatjet.delta_r(higgs[:, 0]),
        "fj_genV_dR": fatjet.delta_r(v),
        "fj_genVstar": fatjet.delta_r(v_star),
        "genV_genVstar_dR": v.delta_r(v_star),
    }

    genHVVVars = {
        "fj_isHVV": is_hww,
        "fj_isHVV_Matched": matched_higgs_mask,
        "fj_isHVV_4q": to_label((num_quarks == 4) & (num_leptons == 0)),
        "fj_isHVV_elenuqq": to_label((num_electrons == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_munuqq": to_label((num_muons == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_taunuqq": to_label((num_taus == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_Vlepton": iswlepton,
        "fj_isHVV_Vstarlepton": iswstarlepton,
        "fj_genRes_mass": matched_higgs.mass,
        "fj_nquarks": num_m_quarks,
        "fj_ncquarks": num_m_cquarks,
        "fj_lepinprongs": num_m_leptons,
        "lep_daughters": leptons,
        "all_daus": all_daus,
    }

    lepVars = {
        "lepton_pt": all_daus_flat[leptons].pt,
        "lepton_eta": all_daus_flat[leptons].eta,
        "lepton_phi": all_daus_flat[leptons].phi,
        "lepton_mass": all_daus_flat[leptons].mass,
    }

    quarkVars = {
        "quark_pt": all_daus_flat[all_daus_flat_pdgId <= b_PDGID].pt,
        "quark_eta": all_daus_flat[all_daus_flat_pdgId <= b_PDGID].eta,
        "quark_phi": all_daus_flat[all_daus_flat_pdgId <= b_PDGID].phi,
        "quark_mass": all_daus_flat[all_daus_flat_pdgId <= b_PDGID].mass,
    }

    genVars = {**genVars, **genVVars, **genHVVVars, **lepVars, **quarkVars}

    return genVars


# count the number of quarks inside the AK8 jet, require 3 or 4 quarks.
def count_quarks_in_jets(jet_4vec, gen_parts_eta_phi, delta_r_cut=0.8):
    num_jets = len(jet_4vec)
    num_quarks_in_jets = np.zeros(num_jets, dtype=int)

    for i in range(num_jets):
        jet_eta, jet_phi = jet_4vec[i][1], jet_4vec[i][2]

        quark_eta_phi = gen_parts_eta_phi[i]
        quark_eta, quark_phi = quark_eta_phi[:, 0], quark_eta_phi[:, 1]

        delta_eta = jet_eta - quark_eta
        delta_phi = jet_phi - quark_phi

        delta_r_squared = delta_eta**2 + delta_phi**2
        quarks_in_jet = np.sqrt(delta_r_squared) < delta_r_cut

        num_quarks_in_jets[i] = np.sum(quarks_in_jet)

    return num_quarks_in_jets


def dRcleanup(events_final, GenlepVars):

    # fj_idx_lep = ak.argmin(events_final.FatJet.delta_r(candidatelep_p4), axis=1, keepdims=True)
    # candidatefj = ak.firsts(events_final.FatJet[fj_idx_lep])

    higgs = events_final.GenPart[(abs(events_final.GenPart.pdgId) == HIGGS_PDGID) * events_final.GenPart.hasFlags(GEN_FLAGS)]
    HWWidx = ak.argmin(events_final.FatJet.delta_r(ak.firsts(higgs)), axis=1, keepdims=True)

    # Get FatJetPFCands 4-vector, up to 150 length to suit the input of Oz's function
    HWW_FatJetPFCands = events_final.FatJetPFCands.jetIdx == ak.firsts(HWWidx)
    HWW_FatJetPFCands_pFCandsIdx = events_final.FatJetPFCands.pFCandsIdx[HWW_FatJetPFCands]

    pt_array = ak.Array(events_final.PFCands.pt)
    eta_array = ak.Array(events_final.PFCands.eta)
    phi_array = ak.Array(events_final.PFCands.phi)
    mass_array = ak.Array(events_final.PFCands.mass)

    # Need to clean PFCands with dR(l,pf)<0.2
    lep_eta = GenlepVars["GenlepEta"]
    lep_phi = GenlepVars["GenlepPhi"]

    # this is because the length of PFCands can be up to 409, so we pad to target = 500
    pf_eta = pad_val(eta_array, target=500, axis=1, value=0)
    pf_phi = pad_val(phi_array, target=500, axis=1, value=0)
    pf_pt = pad_val(pt_array, target=500, axis=1, value=0)
    pf_mass = pad_val(mass_array, target=500, axis=1, value=0)

    lep_eta_reshaped = lep_eta.reshape(-1, 1)
    lep_phi_reshaped = lep_phi.reshape(-1, 1)

    delta_eta = lep_eta_reshaped - pf_eta
    delta_phi = lep_phi_reshaped - pf_phi

    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

    pf_eta_rm_lep = np.copy(pf_eta)
    pf_phi_rm_lep = np.copy(pf_phi)
    pf_pt_rm_lep = np.copy(pf_pt)
    pf_mass_rm_lep = np.copy(pf_mass)

    pf_eta_rm_lep[delta_r < 0.2] = 0.0
    pf_phi_rm_lep[delta_r < 0.2] = 0.0
    pf_pt_rm_lep[delta_r < 0.2] = 0.0
    pf_mass_rm_lep[delta_r < 0.2] = 0.0

    selected_eta = ak.Array(pf_eta_rm_lep)[HWW_FatJetPFCands_pFCandsIdx]
    selected_phi = ak.Array(pf_phi_rm_lep)[HWW_FatJetPFCands_pFCandsIdx]
    selected_pt = ak.Array(pf_pt_rm_lep)[HWW_FatJetPFCands_pFCandsIdx]
    selected_mass = ak.Array(pf_mass_rm_lep)[HWW_FatJetPFCands_pFCandsIdx]

    # pad the selected 4-vec array up to length of 150 to match the Lund Plane input
    selected_pt_padded = pad_val(selected_pt, 150, 0, 1, True)
    selected_eta_padded = pad_val(selected_eta, 150, 0, 1, True)
    selected_phi_padded = pad_val(selected_phi, 150, 0, 1, True)
    selected_mass_padded = pad_val(selected_mass, 150, 0, 1, True)

    pf_cands_px = selected_pt_padded * np.cos(selected_phi_padded)
    pf_cands_py = selected_pt_padded * np.sin(selected_phi_padded)
    pf_cands_pz = selected_pt_padded * np.sinh(selected_eta_padded)
    pf_cands_E = np.sqrt(pf_cands_px**2 + pf_cands_py**2 + pf_cands_pz**2 + selected_mass_padded**2)

    pf_cands_pxpypzE_lvqq = np.dstack((pf_cands_px, pf_cands_py, pf_cands_pz, pf_cands_E))
    return pf_cands_pxpypzE_lvqq


import sys

sys.path.insert(0, "")
sys.path.append("LundReweighting")
sys.path.append("LundReweighting/utils")
import ROOT

from boostedhiggs.LundReweighting.utils import LundReweighter

# # from utils.LundReweighter import *
# # from utils.Utils import *
# from LundReweighter import LundReweighter


def getLPweights(events, candidatefj):
    """
    Relies on
        (1) higgs_jet_4vec_Hlvqq
        (2) gen_parts_eta_phi_Hlvqq_2q
        (3) pf_cands_pxpypzE_lvqq
    """

    print("events", len(events))
    genVars = match_H(events.GenPart, candidatefj)

    higgs_jet_4vec_Hlvqq = np.array(
        np.stack(
            (np.array(candidatefj.pt), np.array(candidatefj.eta), np.array(candidatefj.phi), np.array(candidatefj.mass)),
            axis=1,
        )  # four vector for HWW jet
    )

    skim_vars = {
        "eta": "Eta",
        "phi": "Phi",
        "mass": "Mass",
        "pt": "Pt",
    }

    Gen2qVars = {
        f"Gen2q{var}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(genVars[f"quark_{key}"], 2, axis=1, clip=True),
                FILL_NONE_VALUE,
            )
        )
        for key, var in skim_vars.items()
    }

    GenlepVars = {
        f"Genlep{var}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(genVars[f"lepton_{key}"], 1, axis=1, clip=True),
                FILL_NONE_VALUE,
            )
        )
        for key, var in skim_vars.items()
    }

    # prepare eta, phi array only for 2q, used for Lund Plane reweighting
    # since it only takes quarks gen-level 4-vector as input
    eta_2q = Gen2qVars["Gen2qEta"]
    phi_2q = Gen2qVars["Gen2qPhi"]
    gen_parts_eta_phi_Hlvqq_2q = np.array(np.dstack((eta_2q, phi_2q)))

    # prepare eta, phi array for 2q + lep, to do gen-matching of the jet
    eta = np.concatenate([Gen2qVars["Gen2qEta"], GenlepVars["GenlepEta"]], axis=1)
    phi = np.concatenate([Gen2qVars["Gen2qPhi"], GenlepVars["GenlepPhi"]], axis=1)

    gen_parts_eta_phi_HWW = np.array(np.dstack((eta, phi)))
    LPnumquarks = count_quarks_in_jets(higgs_jet_4vec_Hlvqq, gen_parts_eta_phi_HWW)
    # Hlvqq_cut = LPnumquarks >= 3

    pf_cands_pxpypzE_lvqq = dRcleanup(events, GenlepVars)

    # gen_parts_eta_phi_Hlvqq_2q = gen_parts_eta_phi_Hlvqq_2q[Hlvqq_cut]
    # pf_cands_pxpypzE_lvqq = pf_cands_pxpypzE_lvqq[Hlvqq_cut]
    # higgs_jet_4vec_Hlvqq = higgs_jet_4vec_Hlvqq[Hlvqq_cut]
    # candidatefj = candidatefj[Hlvqq_cut]

    # Input file
    f_ratio_name = "boostedhiggs/LundReweighting/data/ratio_2018.root"
    f_ratio = ROOT.TFile.Open(f_ratio_name)

    # nominal data/MC Lund plane ratio (3d histogram)
    h_ratio = f_ratio.Get("ratio_nom")
    # systematic variations
    h_ratio_sys_up = f_ratio.Get("ratio_sys_tot_up")
    h_ratio_sys_down = f_ratio.Get("ratio_sys_tot_down")

    # directory of pt extrapolation fits
    f_ratio.cd("pt_extrap")
    rdir = ROOT.gDirectory  # get the present working directory and give it to rdir

    # Main class for reweighting utilities
    LP_rw = LundReweighter(pt_extrap_dir=rdir)

    max_evts = len(pf_cands_pxpypzE_lvqq)
    print("max_evts", max_evts)

    # Compute reweighting factors

    # PF candidates in the AK8 jet
    pf_cands = pf_cands_pxpypzE_lvqq[:max_evts]

    # Generator level quarks from hard process
    gen_parts_eta_phi = gen_parts_eta_phi_Hlvqq_2q[:max_evts]

    # ak8_jets = d.get_masked('jet_kinematics')[:max_evts][:,2:6].astype(np.float64)
    ak8_jets = higgs_jet_4vec_Hlvqq[:max_evts]

    # Nominal event weights of the MC, assume every event is weight '1' for this example
    weights_nom = np.ones(max_evts)

    LP_weights = []
    LP_weights_sys_up = []
    LP_weights_sys_down = []
    stat_smeared_weights = []
    pt_smeared_weights = []

    # Number of toys for statistical and pt extrapolation uncertainties
    nToys = 100
    # Noise vectors used to to generate the toys
    # NOTE the same vector has to be used for the whole sample/signal file for the toys to be consistent
    rand_noise = np.random.normal(size=(nToys, h_ratio.GetNbinsX(), h_ratio.GetNbinsY(), h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size=(nToys, h_ratio.GetNbinsY(), h_ratio.GetNbinsZ(), 3))

    for i, cands in enumerate(pf_cands):
        # Get the subjets, splittings and checking matching based on PF candidates in the jet and gen-level quarks
        subjets, splittings, _, _ = LP_rw.get_splittings_and_matching(cands, gen_parts_eta_phi[i], ak8_jets[i])
        # Gets the nominal LP reweighting factor for this event and statistical + pt extrapolation toys
        LP_weight, stat_smeared_weight, pt_smeared_weight = LP_rw.reweight_lund_plane(
            h_rw=h_ratio,
            subjets=subjets,
            splittings=splittings,
            rand_noise=rand_noise,
            pt_rand_noise=pt_rand_noise,
        )
        # Now get systematic variations
        LP_weight_sys_up, _, _ = LP_rw.reweight_lund_plane(h_rw=h_ratio_sys_up, subjets=subjets, splittings=splittings)
        LP_weight_sys_down, _, _ = LP_rw.reweight_lund_plane(h_rw=h_ratio_sys_down, subjets=subjets, splittings=splittings)

        LP_weights.append(LP_weight)
        stat_smeared_weights.append(stat_smeared_weight)
        pt_smeared_weights.append(pt_smeared_weight)

        LP_weights_sys_up.append(LP_weight_sys_up)
        LP_weights_sys_down.append(LP_weight_sys_down)

    # Normalize weights to preserve normalization of the MC sample

    # The nominal Lund Plane correction event weights
    LP_weights = LP_rw.normalize_weights(LP_weights) * weights_nom

    # Toy variations for stat and pt uncertainties
    stat_smeared_weights = LP_rw.normalize_weights(stat_smeared_weights) * weights_nom.reshape(max_evts, 1)
    pt_smeared_weights = LP_rw.normalize_weights(pt_smeared_weights) * weights_nom.reshape(max_evts, 1)

    # Systematic up/down variations
    LP_weights_sys_up = LP_rw.normalize_weights(LP_weights_sys_up) * weights_nom
    LP_weights_sys_down = LP_rw.normalize_weights(LP_weights_sys_down) * weights_nom

    f_ratio.Close()
    return LP_weights, LP_weights_sys_up, LP_weights_sys_down, LPnumquarks
