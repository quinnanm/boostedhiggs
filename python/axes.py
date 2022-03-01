import hist as hist2
axis_dict = {
    'lepton_pt': hist2.axis.Regular(50, 20, 500, name='var', label=r'Lepton $p_T$ [GeV]'),
    'lep_isolation': hist2.axis.Regular(20, 0, 3.5, name='var', label=r'Lepton iso'),
    'ht':  hist2.axis.Regular(20, 180, 1500, name='var', label='HT [GeV]'),
    'dr_jet_candlep': hist2.axis.Regular(15, 0, 0.8, name='var', label=r'$\Delta R(l, Jet)$'),
    'met':  hist2.axis.Regular(50, 0, 400, name='var', label='MET [GeV]'),
    'mt_lep_met':  hist2.axis.Regular(20, 0, 300, name='var', label=r'$m_T(lep, p_T^{miss})$ [GeV]'),
    'mu_mvaId': hist2.axis.Regular(20, -1, 1, name='var', label='Muon MVAID'),
    'leadingfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'Jet $p_T$ [GeV]'),
    'leadingfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'Jet $m_{sd}$ [GeV]'),
    'secondfj_pt': hist2.axis.Regular(30, 200, 1000, name='var', label=r'2nd Jet $p_T$ [GeV]'),
    'secondfj_msoftdrop': hist2.axis.Regular(30, 0, 200, name='var', label=r'2nd Jet $m_{sd}$ [GeV]'),
    'bjets_ophem_leadingfj': hist2.axis.Regular(15, 0, 0.4, name='var', label=r'btagFlavB (opphem)'),
    'bjets_ophem_lepfj': hist2.axis.Regular(15, 0, 0.4, name='var', label=r'btagFlavB (opphem)'),
}
