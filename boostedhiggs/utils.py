import numpy as np
from coffea.analysis_tools import PackedSelection
import awkward as ak


def getParticles(genparticles, lowid=22, highid=25, flags=['fromHardProcess', 'isLastCopy']):
    """
    returns the particle objects that satisfy a low id,
    high id condition and have certain flags
    """
    absid = abs(genparticles.pdgId)
    return genparticles[
        ((absid >= lowid) & (absid <= highid))
        & genparticles.hasFlags(flags)
    ]


def match_HWW(genparticles, candidatefj):
    """
    return the number of matched objects (hWW*),daughters,
    and gen flavor (enuqq, munuqq, taunuqq)
    """
    higgs = getParticles(genparticles, 25)   # genparticles is the full set... this function selects Higgs particles
    is_hWW = ak.all(abs(higgs.children.pdgId) == 24, axis=2)    # W~24 so we get H->WW (limitation: only picking one W and assumes the other will be there)

    higgs = higgs[is_hWW]
    higgs_wstar = higgs.children[ak.argmin(higgs.children.mass, axis=2, keepdims=True)]
    higgs_w = higgs.children[ak.argmax(higgs.children.mass, axis=2, keepdims=True)]

    prompt_electron = getParticles(genparticles, 11, 11, ['isPrompt', 'isLastCopy'])    # isPrompt avoids displaced leptons
    prompt_muon = getParticles(genparticles, 13, 13, ['isPrompt', 'isLastCopy'])
    prompt_tau = getParticles(genparticles, 15, 15, ['isPrompt', 'isLastCopy'])
    prompt_q = getParticles(genparticles, 0, 5, ['fromHardProcess', 'isLastCopy'])      # 0-5 not 0-6 to avoid top quark
    prompt_q = prompt_q[abs(prompt_q.distinctParent.pdgId) == 24]       # parent W

    dr_fj_quarks = candidatefj.delta_r(prompt_q)
    dr_fj_electrons = candidatefj.delta_r(prompt_electron)
    dr_fj_muons = candidatefj.delta_r(prompt_muon)
    dr_fj_taus = candidatefj.delta_r(prompt_tau)
    dr_daughters = ak.concatenate([dr_fj_quarks, dr_fj_electrons, dr_fj_muons, dr_fj_taus], axis=1)
    hWW_nprongs = ak.sum(dr_daughters < 0.8, axis=1)   # impose that something must be inside the cone... tells you # of particles from Higgs matched to the jet

    n_electrons = ak.sum(prompt_electron.pt > 0, axis=1)
    n_muons = ak.sum(prompt_muon.pt > 0, axis=1)
    n_taus = ak.sum(prompt_tau.pt > 0, axis=1)
    n_quarks = ak.sum(prompt_q.pt > 0, axis=1)

    # 4(elenuqq),6(munuqq),8(taunuqq)
    hWW_flavor = (n_quarks == 2) * 1 + (n_electrons == 1) * 3 + (n_muons == 1) * 5 + (n_taus == 1) * 7 + (n_quarks == 4) * 11

    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)    # choose higgs closest to fj
    matchedW = candidatefj.nearest(higgs_w, axis=1, threshold=0.8)  # choose W closest to fj
    matchedWstar = candidatefj.nearest(higgs_wstar, axis=1, threshold=0.8)  # choose Wstar closest to fj

    # 1 (H only), 4(W), 6(W star), 9(H, W and Wstar)
    hWW_matched = (
        (ak.sum(matchedH.pt > 0, axis=1) == 1) * 1
        + (ak.sum(ak.flatten(matchedW.pt > 0, axis=2), axis=1) == 1) * 3
        + (ak.sum(ak.flatten(matchedWstar.pt > 0, axis=2), axis=1) == 1) * 5
    )

    # leptons matched
    dr_fj_leptons = ak.concatenate([dr_fj_electrons, dr_fj_muons], axis=1)

    leptons = ak.concatenate([prompt_electron, prompt_muon], axis=1)
    leptons = leptons[dr_fj_leptons < 0.8]

    # leptons coming from W or W*
    leptons_mass = ak.firsts(leptons.distinctParent.mass)   # # TODO: why need firsts
    higgs_w_mass = ak.firsts(ak.flatten(higgs_w.mass))[ak.firsts(leptons.pt > 0)]
    higgs_wstar_mass = ak.firsts(ak.flatten(higgs_wstar.mass))[ak.firsts(leptons.pt > 0)]

    iswlepton = (leptons_mass == higgs_w_mass)
    iswstarlepton = (leptons_mass == higgs_wstar_mass)

    ret = {"hWW_flavor": hWW_flavor,
           "hWW_matched": hWW_matched,
           "hWW_nprongs": hWW_nprongs,
           "matchedH": matchedH,
           "iswlepton": iswlepton,
           "iswstarlepton": iswstarlepton}

    return ret


def match_Htt(genparticles, candidatefj):
    higgs = getParticles(genparticles, 25)
    is_htt = ak.all(abs(higgs.children.pdgId) == 15, axis=2)

    higgs = higgs[is_htt]

    fromtau_electron = getParticles(events.GenPart, 11, 11, ['isDirectTauDecayProduct'])
    fromtau_muon = getParticles(events.GenPart, 13, 13, ['isDirectTauDecayProduct'])
    tau_visible = events.GenVisTau

    n_visibletaus = ak.sum(tau_visible.pt > 0, axis=1)
    n_electrons_fromtaus = ak.sum(fromtau_electron.pt > 0, axis=1)
    n_muons_fromtaus = ak.sum(fromtau_muon.pt > 0, axis=1)
    # 3(elenuqq),6(munuqq),8(taunuqq)
    htt_flavor = (n_quarks == 2) * 1 + (n_electrons == 1) * 3 + (n_muons == 1) * 5 + (n_taus == 1) * 7

    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)
    dr_fj_visibletaus = candidatefj.delta_r(tau_visible)
    dr_fj_electrons = candidatefj.delta_r(fromtau_electron)
    dr_fj_muons = candidatefj.delta_r(fromtau_muon)
    dr_daughters = ak.concatenate([dr_fj_visibletaus, dr_fj_electrons, dr_fj_muons], axis=1)
    # 1 (H only), 4 (H and one tau/electron or muon from tau), 5 (H and 2 taus/ele)
    htt_matched = (ak.sum(matchedH.pt > 0, axis=1) == 1) * 1 + (ak.sum(dr_daughters < 0.8, axis=1) == 1) * 3 + (ak.sum(dr_daughters < 0.8, axis=1) == 2) * 5

    return htt_flavor, htt_matched, matchedH, higgs


def pad_val(
    arr: ak.Array,
    target: int,
    value: float,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
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
