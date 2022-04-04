import correctionlib
import awkward as ak
import numpy as np
import pickle as pkl
import importlib.resources

from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
btagWPs = {
    "deepJet": {
        '2016': {  # this one is preVFP
            'L': 0.0508,
            'M': 0.2598,
            'T': 0.6502,
        },
        '2016postVFP': {
            'L': 0.0480,
            'M': 0.2489,
            'T': 0.6377,
        },
        '2017': {
            'L': 0.0532,
            'M': 0.3040,
            'T': 0.7476,
        },
        '2018': {
            'L': 0.0490,
            'M': 0.2783,
            'T': 0.7100,
        },
    }
}
taggerBranch = {
    "deepJet": "btagDeepFlavB",
    "deepCSV": "btagDeep"
}


class BTagCorrector:
    def __init__(self, wp, tagger="deepJet", year="2017", mod=""):
        self._year = year + mod
        self._tagger = tagger
        self._wp = wp
        self._btagwp = btagWPs[tagger][year + mod][wp]
        self._branch = taggerBranch[tagger]

        # more docs at https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_201*_UL.html
        self._cset = correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/{year}_UL/btagging.json.gz")

        # efficiency lookup
        with importlib.resources.path("boostedhiggs.data", f"btageff_{tagger}_{wp}_{year}.pkl") as path:
            with open(path, 'rb') as f:
                eff = pkl.load(f)

        self.efflookup = dense_lookup(eff.values(), [ax.edges for ax in eff.axes])

    def lighttagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_incl" % self._tagger].evaluate(syst, self._wp, np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
        return ak.unflatten(sf, nj)

    def btagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_comb" % self._tagger].evaluate(syst, self._wp, np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
        return ak.unflatten(sf, nj)

    def addBtagWeight(self, jets, weights):
        """
        Adding one common multiplicative SF (including bcjets + lightjets)
        weights: weights class from coffea
        jets: jets selected in your analysis
        """
        lightJets = jets[jets.hadronFlavour == 0]
        bcJets = jets[jets.hadronFlavour > 0]

        lightEff = self.efflookup(lightJets.pt, abs(lightJets.eta), lightJets.hadronFlavour)
        bcEff = self.efflookup(bcJets.pt, abs(bcJets.eta), bcJets.hadronFlavour)

        lightPass = lightJets[self._branch] > self._btagwp
        bcPass = bcJets[self._branch] > self._btagwp

        def combine(eff, sf, passbtag):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = ak.prod(sf[passbtag], axis=-1)
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff))[~passbtag], axis=-1)
            return tagged_sf * untagged_sf

        lightweight = combine(
            lightEff,
            self.lighttagSF(lightJets, "central"),
            lightPass
        )
        bcweight = combine(
            bcEff,
            self.btagSF(bcJets, "central"),
            bcPass
        )

        # nominal weight = btagSF (btagSFbc*btagSFlight)
        nominal = lightweight * bcweight
        weights.add('btagSF', lightweight * bcweight)

        # systematics:
        # btagSFlight_{year}: btagSFlight_up/down
        # btagSFbc_{year}: btagSFbc_up/down
        # btagSFlight_correlated: btagSFlight_up/down_correlated
        # btagSFbc_correlated:  btagSFbc_up/down_correlated
        weights.add(
            'btagSFlight_%s' % self._year,
            np.ones(len(nominal)),
            weightUp=combine(
                lightEff,
                self.lighttagSF(lightJets, "up"),
                lightPass
            ),
            weightDown=combine(
                lightEff,
                self.lighttagSF(lightJets, "down"),
                lightPass
            )
        )
        weights.add(
            'btagSFbc_%s' % self._year,
            np.ones(len(nominal)),
            weightUp=combine(
                bcEff,
                self.btagSF(bcJets, "up"),
                bcPass
            ),
            weightDown=combine(
                bcEff,
                self.btagSF(bcJets, "down"),
                bcPass
            )
        )
        weights.add(
            'btagSFlight_correlated',
            np.ones(len(nominal)),
            weightUp=combine(
                lightEff,
                self.lighttagSF(lightJets, "up_correlated"),
                lightPass
            ),
            weightDown=combine(
                lightEff,
                self.lighttagSF(lightJets, "down_correlated"),
                lightPass
            )
        )
        weights.add(
            'btagSFbc_correlated',
            np.ones(len(nominal)),
            weightUp=combine(
                bcEff,
                self.btagSF(bcJets, "up_correlated"),
                bcPass
            ),
            weightDown=combine(
                bcEff,
                self.btagSF(bcJets, "down_correlated"),
                bcPass
            )
        )
        return nominal


if __name__ == '__main__':
    from coffea.analysis_tools import Weights
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    tagger = "deepJet"
    year = "2018"
    wp = "M"

    tt_files = {
        "2018": "root://cmsxrootd-site.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/0520A050-AF68-EF43-AA5B-5AA77C74ED73.root",
    }

    events = NanoEventsFactory.from_root(
        tt_files[year],
        entry_stop=100_000,
        schemaclass=NanoAODSchema,
    ).events()

    phasespace_cuts = (
        (events.Jet.pt > 30)
        & (abs(events.Jet.eta) < 2.5)
        & events.Jet.isTight
        & (events.Jet.puId > 0)
    )

    # efficiency is specific for analysis: https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#b_tagging_efficiency_in_MC_sampl
    jets = ak.flatten(events.Jet[phasespace_cuts])
    import hist
    efficiencyinfo = (
        hist.Hist.new
        .Reg(20, 40, 300, name="pt")
        .Reg(4, 0, 2.5, name="abseta")
        .IntCat([0, 4, 5], name="flavor")
        .Bool(name="passWP")
        .Double()
        .fill(
            pt=jets.pt,
            abseta=abs(jets.eta),
            flavor=jets.hadronFlavour,
            passWP=jets[taggerBranch[tagger]] > btagWPs[tagger][year][wp]
        )
    )
    eff = efficiencyinfo[{"passWP": True}] / efficiencyinfo[{"passWP": sum}]

    with open(f'data/btageff_{tagger}_{wp}_{year}.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(eff, f)

    jets = events.Jet[phasespace_cuts]
    weights = Weights(len(events), storeIndividual=True)
    b = BTagCorrector('M', tagger, '2018')
    nom = b.addBtagWeight(jets, weights)
    print(nom)
