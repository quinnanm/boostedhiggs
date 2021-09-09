import os
import argparse
import _pickle as cPickle
import gc
import time
import json
import copy

import hist as hist2
from coffea import processor

# hists path
#path = "/eos/uscms/store/user/docampoh/boostedhiggs/Aug31_UL/outfiles/" 
path = "/eos/uscms/store/user/cmantill/boostedhiggs/Sep6_UL/outfiles/"

# xsec paths
xsec_path = "fileset/xsecs.json"
#xsec_path = "fileset/xsecs_dylan.json"

def iter_flatten(iterable):
    """flatten nested lists"""
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e

def read_hists(sample):
    files = os.listdir(path)

    # make sample dictionary
    # each key must be different for samples with different cross section
    # must also match the sample in xsecs.json..?
    hww = {
        "GluGluHToWWToLNuQQ": [file for file in files if "GluGluHToWWToLNuQQ" in file],
    }
    tt = {
        "TTToSemiLeptonic": [file for file in files if "TTToSemiLeptonic" in file],
        "TTToHadronic": [file for file in files if "TTToHadronic" in file],
        "TTTo2L2Nu": [file for file in files if "TTTo2L2Nu" in file],
    }
    # TODO: add st when xsections are included
    # st = {
    #     "ST_s-channel_4f": [file for file in files if "ST_s-channel_4f" in file],
    #     "ST_tW_antitop_5f": [file for file in files if "ST_tW_antitop_5f" in file],
    #     "ST_tW_top_5f": [file for file in files if "ST_tW_top_5f" in file],
    #     "ST_t-channel_muDecays": [file for file in files if "ST_t-channel_muDecays" in file],
    #     "ST_t-channel_eleDecays": [file for file in files if "ST_t-channel_eleDecays" in file],
    #     "ST_t-channel_antitop_5f": [file for file in files if "ST_t-channel_antitop_5f" in file],
    # }
    qcd = {
        "QCD_HT300to500": [file for file in files if "QCD_HT300to500" in file],
        "QCD_HT500to700": [file for file in files if "QCD_HT500to700" in file],
        "QCD_HT700to1000": [file for file in files if "QCD_HT700to1000" in file],
        "QCD_HT1000to1500": [file for file in files if "QCD_HT1000to1500" in file],
        "QCD_HT1500to2000": [file for file in files if "QCD_HT1500to2000" in file],
        "QCD_HT2000toInf": [file for file in files if "QCD_HT2000toInf" in file]
    }
    wjets = {
        "WJetsToLNu_HT-200To400": [file for file in files if "WJetsToLNu_HT-200To400" in file],
        "WJetsToLNu_HT-400To600": [file for file in files if "WJetsToLNu_HT-400To600" in file],
        "WJetsToLNu_HT-600To800": [file for file in files if "WJetsToLNu_HT-600To800" in file],
        "WJetsToLNu_HT-800To1200": [file for file in files if "WJetsToLNu_HT-800To1200" in file],
        "WJetsToLNu_HT-1200To2500": [file for file in files if "WJetsToLNu_HT-1200To2500" in file],
    }
    # TODO: add dyjets
    # dyjets = {        
    # }
    singleElectron = {
        "SingleElectron": [file for file in files if "SingleElectron" in file]
    }
    singleMuon = {
        "SingleMuon": [file for file in files if "SingleMuon" in file]
    }
    
    samples_dics = {
        "hww": hww,
        "tt": tt,
        "qcd": qcd,
        "wjets": wjets,
        "electron": singleElectron,
        "muon": singleMuon,
    }

    return samples_dics[sample]



def load_hists(sample_dic, histograms):
    """load and accumulate histograms by sample"""
    
    hists = {key: [] for key in sample_dic}
    
    for key, hist in sample_dic.items():
        print(f"processing {key} histograms")
        for h in hist:
            with open(f"{path}/{h}", "rb") as f:
                # disable garbage collector
                gc.disable()
                
                # load and save histograms
                H = cPickle.load(f)
                k = [key for key in H]
                histos = {key:val for key, val in H[k[0]].items() if key in histograms}
                hists[key].append(histos)
         
                # enable garbage collector again
                gc.enable()

    for key in sample_dic:
        sample_dic[key] = processor.accumulate(hists[key])

    return sample_dic


def scale_hists(sample_dic, lumi):
    """scale histograms to cross section"""

    with open(xsec_path) as f:
        xsecs = json.load(f)

    out = []
    for sample in sample_dic:
    
        hists = sample_dic[sample]
        sumw = sample_dic[sample]["sumw"]
        
        try:
            xsec = eval(xsecs[sample])
        except:
            xsec = xsecs[sample]

        weight = (xsec * lumi) / sumw
        hists = copy.deepcopy(hists)

        for h in hists.values():
            if isinstance(h, hist2.Hist):
                h *= weight
                
        out.append(hists)
    
    if len(out) == 1:
        return {sample.split("_")[0]: out[0]}
    else:
        return {sample.split("_")[0]: processor.accumulate(out)}


def main(args):
    histograms = [hist for hist in iter_flatten(args.histogram)]
    print(f"histogram: {histograms[1:][0]}")
    
    start_time = time.time()
    sample_dic = read_hists(args.sample)
    sample_dic = load_hists(sample_dic, histograms)
    if args.sample not in ["electron", "muon"]:
        output = scale_hists(sample_dic, 41500)
    else:
        output = sample_dic

    output_path = "hists/"
    output_name = ""
    for hist in histograms[1:]:
        output_name += "_" + hist

    os.system(f"mkdir -p {output_path}/{args.histogram[-1][0]}")
    with open(f"{output_path}/{args.histogram[-1][0]}/{args.sample}{output_name}.pkl", "wb") as f:
        cPickle.dump(output, f, protocol=-1)
    
    end_time = time.time()

    print(f"{histograms[1:][0]} histograms processed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", dest="sample", type=str)
    parser.add_argument("--histogram", dest="histogram", nargs="+", action="append", default=["sumw"], type=str)
    parser.add_argument("--lumi", dest="lumi", default=41500, type=float)
    args = parser.parse_args()

    main(args)
