from typing import Union

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

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

Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

PI_PDGID = 211
PO_PDGID = 221
PP_PDGID = 111

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]

FILL_NONE_VALUE = -99999

JET_DR = 0.8


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

    if type(pdgids) == list:
        mask = gen_pdgids == pdgids[0]
        for pdgid in pdgids[1:]:
            mask = mask | (gen_pdgids == pdgid)
    else:
        mask = gen_pdgids == pdgids

    return ak.all(mask, axis=ax) if byall else mask


def to_label(array: ak.Array) -> ak.Array:
    return ak.values_astype(array, np.int32)


def match_H(genparts: GenParticleArray, fatjet: FatJetArray, selection=None, dau_pdgid=W_PDGID):
    """Gen matching for Higgs samples"""
    higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # only select events that match an specific decay
    signal_mask = ak.firsts(ak.all(abs(higgs.children.pdgId) == dau_pdgid, axis=2))

    matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)][:, 0]
    matched_higgs_children = matched_higgs.children

    genVars = {"fj_genH_pt": ak.fill_none(matched_higgs.pt, FILL_NONE_VALUE)}

    if dau_pdgid == W_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [W_PDGID], byall=False)
        matched_higgs_children = matched_higgs_children[children_mask]

        # order by mass, select lower mass child as V* and higher as V
        children_mass = matched_higgs_children.mass
        v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=1, keepdims=True)])
        v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=1, keepdims=True)])

        is_hww_matched = ak.any(children_mask, axis=1)

        genVVars = {
            "fj_genV_dR": fatjet.delta_r(v),
            "fj_genVstar": fatjet.delta_r(v_star),
            "genV_genVstar_dR": v.delta_r(v_star),
        }

        # VV daughters
        daughters = ak.flatten(matched_higgs_children.distinctChildren, axis=2)
        daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
        daughters_pdgId = abs(daughters.pdgId)

        # exclude neutrinos from nprongs count
        daughters_nov = daughters[
            ((daughters_pdgId != vELE_PDGID) & (daughters_pdgId != vMU_PDGID) & (daughters_pdgId != vTAU_PDGID))
        ]
        nprongs = ak.sum(fatjet.delta_r(daughters_nov) < JET_DR, axis=1)

        # get tau decays
        taudaughters = daughters[(daughters_pdgId == TAU_PDGID)].children
        taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        taudecay = (
            # pions/kaons (hadronic tau) * 1
            (
                ak.sum(
                    (taudaughters_pdgId == ELE_PDGID) | (taudaughters_pdgId == MU_PDGID),
                    axis=1,
                )
                == 0
            )
            * 1
            # 1 electron * 3
            + (ak.sum(taudaughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(taudaughters_pdgId == MU_PDGID, axis=1) == 1) * 5
        )
        # flatten taudecay - so painful
        taudecay = ak.sum(taudecay, axis=-1)

        # lepton daughters
        lepdaughters = daughters[
            ((daughters_pdgId == ELE_PDGID) | (daughters_pdgId == MU_PDGID) | (daughters_pdgId == TAU_PDGID))
        ]
        lepinprongs = ak.sum(fatjet.delta_r(lepdaughters) < JET_DR, axis=1)  # should be 0 or 1

        lepton_parent = ak.firsts(lepdaughters[fatjet.delta_r(lepdaughters) < JET_DR].distinctParent)
        lepton_parent_mass = lepton_parent.mass

        iswlepton = lepton_parent_mass == v.mass
        iswstarlepton = lepton_parent_mass == v_star.mass

        decay = (
            # 2 quarks * 1
            (ak.sum(daughters_pdgId <= b_PDGID, axis=1) == 2) * 1
            # 1 electron * 3
            + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # 1 tau * 7
            + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
            # 4 quarks * 11
            + (ak.sum(daughters_pdgId <= b_PDGID, axis=1) == 4) * 11
        )

        # print("higgs pt ", matched_higgs.pt[selection][decay[selection] != 6])
        # print("w mass ", v.mass[selection][decay[selection] != 6])
        # print("distinct children ", matched_higgs_children.distinctChildren.pdgId[selection][decay[selection] != 6])
        # print("decay ", decay[selection][decay[selection] != 6])
        # print("dau id ", daughters_pdgId[selection][decay[selection] != 6])
        # print(" ")

        # number of c quarks in V decay inside jet
        cquarks = daughters_nov[abs(daughters_nov.pdgId) == c_PDGID]
        ncquarks = ak.sum(fatjet.delta_r(cquarks) < JET_DR, axis=1)

        genHVVVars = {
            "fj_nprongs": nprongs,
            "fj_ncquarks": ncquarks,
            "fj_lepinprongs": lepinprongs,
            "fj_H_VV_4q": to_label(decay == 11),
            "fj_H_VV_elenuqq": to_label(decay == 4),
            "fj_H_VV_munuqq": to_label(decay == 6),
            "fj_H_VV_leptauelvqq": to_label((decay == 8) & (taudecay == 3)),
            "fj_H_VV_leptaumuvqq": to_label((decay == 8) & (taudecay == 5)),
            "fj_H_VV_hadtauvqq": to_label((decay == 8) & (taudecay == 1)),
            "fj_H_VV_isVlepton": iswlepton,
            "fj_H_VV_isVstarlepton": iswstarlepton,
            "fj_H_VV_isMatched": is_hww_matched,
        }

        genVars = {**genVars, **genVVars, **genHVVVars}

    elif dau_pdgid == TAU_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [TAU_PDGID], byall=False)
        daughters = matched_higgs_children[children_mask]

        is_htt_matched = ak.any(children_mask, axis=1)

        taudaughters = daughters[(abs(daughters.pdgId) == TAU_PDGID)].children
        taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        taudaughters = taudaughters[
            ((taudaughters_pdgId != vELE_PDGID) & (taudaughters_pdgId != vMU_PDGID) & (taudaughters_pdgId != vTAU_PDGID))
        ]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        flat_taudaughters_pdgId = ak.flatten(taudaughters_pdgId, axis=2)

        extra_taus = ak.any(taudaughters_pdgId == TAU_PDGID, axis=2)
        children_pdgId = abs(taudaughters[extra_taus].children.pdgId)

        taudecay = (
            # pions/kaons (full hadronic tau) * 1
            (
                (
                    ak.sum(
                        (flat_taudaughters_pdgId == PI_PDGID)
                        | (flat_taudaughters_pdgId == PO_PDGID)
                        | (flat_taudaughters_pdgId == PP_PDGID),
                        axis=1,
                    )
                    > 0
                )
            )
            * 1
            # 1 electron * 3
            + (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # two leptons
            + (
                (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 2)
                | (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 2)
            )
            * 7
        )

        extradecay = (
            (
                (
                    ak.sum(
                        ak.sum(
                            (children_pdgId == PI_PDGID) | (children_pdgId == PO_PDGID) | (children_pdgId == PP_PDGID),
                            axis=-1,
                        ),
                        axis=1,
                    )
                    > 0
                )
            )
            * 1
            + (ak.sum(ak.sum(children_pdgId == ELE_PDGID, axis=-1), axis=1) == 1) * 3
            + (ak.sum(ak.sum(children_pdgId == MU_PDGID, axis=-1), axis=1) == 1) * 5
            + (
                (ak.sum(ak.sum(children_pdgId == MU_PDGID, axis=-1), axis=1) == 2)
                | (ak.sum(ak.sum(children_pdgId == ELE_PDGID, axis=-1), axis=1) == 2)
            )
            * 7
        )
        extradecay = ak.sum(extradecay, axis=-1)

        elehad = ((taudecay == 4) & (extradecay == 0)) | ((extradecay == 4) & (taudecay == 0))
        muhad = ((taudecay == 6) & (extradecay == 0)) | ((extradecay == 6) & (taudecay == 0))
        leplep = ((taudecay == 7) | (taudecay == 8)) | ((extradecay == 7) | (extradecay == 8))
        hadhad = ~elehad & ~muhad & ~leplep

        # to painfully debug
        # np.set_printoptions(threshold=np.inf)
        # print(ak.argsort((is_htt_matched)).to_numpy())
        # print(ak.flatten(taudecay).to_numpy())
        # idx= ak.argsort((is_htt_matched)).to_numpy()
        # idx = [74,2023,2037,2887,3121,3435,3838,4599,4702,4906,5266,5703,6063,6498,6799,7642,8820,8828,8999,9005,9455,9564,
        # 11178,11597,11736,12207,12325,12504,12697,12780,13151,13690]
        # for i in idx:
        #     print(i,flat_taudaughters_pdgId[i],extra_taus[i],children_pdgId[i])
        #     print(elehad[i],muhad[i],leplep[i],hadhad[i])
        #     print(taudecay[i],extradecay[i])

        genHTTVars = {
            "fj_H_tt_hadhad": to_label(hadhad),
            "fj_H_tt_elehad": to_label(elehad),
            "fj_H_tt_muhad": to_label(muhad),
            "fj_H_tt_leplep": to_label(leplep),
            "fj_H_tt_isMatched": is_htt_matched,
        }

        genVars = {**genVars, **genHTTVars}

    # added Feb13 2023
    nmuons = {"n_gen_muons": (ak.sum(daughters_pdgId == MU_PDGID, axis=1))}
    decay = {"decay": decay}

    # a = daughters_pdgId[nmuons["nmuons"] == 0]
    # a = a[~ak.is_none(a)]
    # for i, aa in enumerate(a):
    #     print(aa)
    #     if i == 100:
    #         break

    # test_ = {
    #     "d_PDGID": 1,
    #     "u_PDGID": 2,
    #     "s_PDGID": 3,
    #     "c_PDGID": 4,
    #     "ELE_PDGID": 11,
    #     "vELE_PDGID": 12,
    #     "MU_PDGID": 13,
    #     "vMU_PDGID": 13,
    #     "TAU_PDGID": 13,
    #     "vTAU_PDGID": 13,
    # }
    # npgid = {}
    # for key, val in test_.items():
    #     npgid[key] = ak.sum(daughters_pdgId == val, axis=1)

    genVars = {**genVars, **nmuons, **decay}

    return genVars, signal_mask


def match_V(genparts: GenParticleArray, fatjet: FatJetArray):
    vs = genparts[get_pid_mask(genparts, [W_PDGID, Z_PDGID], byall=False) * genparts.hasFlags(GEN_FLAGS)]
    matched_vs = vs[ak.argmin(fatjet.delta_r(vs), axis=1, keepdims=True)]
    matched_vs_mask = ak.any(fatjet.delta_r(matched_vs) < 0.8, axis=1)

    daughters = ak.flatten(matched_vs.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)
    decay = (
        # 2 quarks * 1
        (ak.sum(daughters_pdgId < b_PDGID, axis=1) == 2) * 1
        # >=1 electron * 3
        + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) >= 1) * 3
        # >=1 muon * 5
        + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) >= 1) * 5
        # >=1 tau * 7
        + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) >= 1) * 7
    )

    daughters_nov = daughters[
        ((daughters_pdgId != vELE_PDGID) & (daughters_pdgId != vMU_PDGID) & (daughters_pdgId != vTAU_PDGID))
    ]
    nprongs = ak.sum(fatjet.delta_r(daughters_nov) < JET_DR, axis=1)

    lepdaughters = daughters[
        ((daughters_pdgId == ELE_PDGID) | (daughters_pdgId == MU_PDGID) | (daughters_pdgId == TAU_PDGID))
    ]
    lepinprongs = 0
    if len(lepdaughters) > 0:
        lepinprongs = ak.sum(fatjet.delta_r(lepdaughters) < JET_DR, axis=1)  # should be 0 or 1

    # number of c quarks
    cquarks = daughters_nov[abs(daughters_nov.pdgId) == c_PDGID]
    ncquarks = ak.sum(fatjet.delta_r(cquarks) < JET_DR, axis=1)

    matched_vdaus_mask = ak.any(fatjet.delta_r(daughters) < 0.8, axis=1)
    matched_mask = matched_vs_mask & matched_vdaus_mask
    genVars = {
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
        "fj_V_isMatched": matched_mask,
        "fj_V_2q": to_label(decay == 1),
        "fj_V_elenu": to_label(decay == 3),
        "fj_V_munu": to_label(decay == 5),
        "fj_V_taunu": to_label(decay == 7),
    }
    return genVars


def match_Top(genparts: GenParticleArray, fatjet: FatJetArray):
    tops = genparts[get_pid_mask(genparts, TOP_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]
    matched_tops = tops[ak.argmin(fatjet.delta_r(tops), axis=1, keepdims=True)]
    matched_tops_mask = ak.any(fatjet.delta_r(matched_tops) < 0.8, axis=1)
    daughters = ak.flatten(matched_tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)

    wboson_daughters = ak.flatten(daughters[(daughters_pdgId == 24)].distinctChildren, axis=2)
    wboson_daughters = wboson_daughters[wboson_daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    wboson_daughters_pdgId = abs(wboson_daughters.pdgId)
    decay = (
        # 2 quarks
        (ak.sum(wboson_daughters_pdgId < b_PDGID, axis=1) == 2) * 1
        # 1 electron * 3
        + (ak.sum(wboson_daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
        # 1 muon * 5
        + (ak.sum(wboson_daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
        # 1 tau * 7
        + (ak.sum(wboson_daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
    )
    bquark = daughters[(daughters_pdgId == 5)]
    matched_b = ak.sum(fatjet.delta_r(bquark) < 0.8, axis=1)

    wboson_daughters_nov = wboson_daughters[
        (
            (wboson_daughters_pdgId != vELE_PDGID)
            & (wboson_daughters_pdgId != vMU_PDGID)
            & (wboson_daughters_pdgId != vTAU_PDGID)
        )
    ]
    # nprongs only includes the number of quarks from W decay (not b!)
    nprongs = ak.sum(fatjet.delta_r(wboson_daughters_nov) < JET_DR, axis=1)

    matched_topdaus_mask = ak.any(fatjet.delta_r(daughters) < 0.8, axis=1)
    matched_mask = matched_tops_mask & matched_topdaus_mask

    # number of c quarks in V decay inside jet
    cquarks = wboson_daughters_nov[abs(wboson_daughters_nov.pdgId) == c_PDGID]
    ncquarks = ak.sum(fatjet.delta_r(cquarks) < JET_DR, axis=1)

    lepdaughters = wboson_daughters[
        (
            (wboson_daughters_pdgId == ELE_PDGID)
            | (wboson_daughters_pdgId == MU_PDGID)
            | (wboson_daughters_pdgId == TAU_PDGID)
        )
    ]

    lepinprongs = 0
    if len(lepdaughters) > 0:
        lepinprongs = ak.sum(fatjet.delta_r(lepdaughters) < JET_DR, axis=1)  # should be 0 or 1

    # get tau decays from V daughters
    taudaughters = wboson_daughters[(wboson_daughters_pdgId == TAU_PDGID)].children
    taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
    taudaughters_pdgId = abs(taudaughters.pdgId)

    taudecay = (
        # pions/kaons (hadronic tau) * 1
        (
            ak.sum(
                (taudaughters_pdgId == ELE_PDGID) | (taudaughters_pdgId == MU_PDGID),
                axis=2,
            )
            == 0
        )
        * 1
        # 1 electron * 3
        + (ak.sum(taudaughters_pdgId == ELE_PDGID, axis=2) == 1) * 3
        # 1 muon * 5
        + (ak.sum(taudaughters_pdgId == MU_PDGID, axis=2) == 1) * 5
    )
    # flatten taudecay - so painful
    taudecay = ak.sum(taudecay, axis=-1)

    genVars = {
        "fj_Top_isMatched": matched_mask,
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
        "fj_Top_bmerged": to_label(matched_b == 1),
        "fj_Top_2q": to_label(decay == 1),
        "fj_Top_elenu": to_label(decay == 3),
        "fj_Top_munu": to_label(decay == 5),
        "fj_Top_hadtauvqq": to_label((decay == 7) & (taudecay == 1)),
        "fj_Top_leptauelvnu": to_label((decay == 7) & (taudecay == 3)),
        "fj_Top_leptaumuvnu": to_label((decay == 7) & (taudecay == 5)),
    }
    return genVars


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


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    signGenWeights: ak.Array,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    selection.add(name, sel)
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up sign of genWeights for MC
        else np.sum(signGenWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def get_neutrino_z(vis, inv, h_mass=125):
    """
    Reconstruct the mass by taking qq jet, lepton and MET
    Then, solve for the z component of the neutrino momentum
    by requiring that the invariant mass of the group of objects is the Higgs mass = 125
    """
    a = h_mass * h_mass - vis.mass * vis.mass + 2 * vis.x * inv.x + 2 * vis.y * inv.y
    A = 4 * (vis.t * vis.t - vis.z * vis.z)
    B = -4 * a * vis.z
    C = 4 * vis.t * vis.t * (inv.x * inv.x + inv.y * inv.y) - a * a
    delta = B * B - 4 * A * C
    neg = -B / (2 * A)
    neg = ak.nan_to_num(neg)
    pos = np.maximum((-B + np.sqrt(delta)) / (2 * A), (-B - np.sqrt(delta)) / (2 * A))
    pos = ak.nan_to_num(pos)

    invZ = (delta < 0) * neg + (delta > 0) * pos
    neutrino = ak.zip(
        {
            "x": inv.x,
            "y": inv.y,
            "z": invZ,
            "t": np.sqrt(inv.x * inv.x + inv.y * inv.y + invZ * invZ),
        },
        with_name="LorentzVector",
    )
    return neutrino
