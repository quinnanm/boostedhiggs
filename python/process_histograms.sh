#!/bin/bash

# path to output
#hists_path="/eos/uscms/store/user/cmantill/boostedhiggs/Sep29_fixedweightsdata_lepmj_UL/outfiles/"
hists_path="/eos/uscms/store/user/cmantill/boostedhiggs/Oct6_fixedweightsdata_lepmj_UL/outfiles"

# path to cross section json 
xsecs_path="/uscms/home/docampoh/nobackup/boostedhiggs/fileset/xsecs.json"

# samples, histograms and regions to read
samples=(hww tt_semileptonic tt_hadronic tt_dileptonic qcd wjets st zjets electron muon)
hists=(jet_kin met_kin lep_kin signal_kin cutflow jet_lep)
regions=(hadel hadmu)

# luminosity
lumi=41500

for hist in "${hists[@]}"; do
    for sample in "${samples[@]}"; do
	echo processing $sample $hist histograms
	if  [[ $sample == muon ]]; then
	    python process_histograms.py --hpath $hists_path --sample $sample --histogram $hist --region hadmu --lumi $lumi --xsecs $xsecs_path
	    continue
	elif [[ $sample == electron ]]; then
	    python process_histograms.py --hpath $hists_path --sample $sample --histogram $hist --region hadel --lumi $lumi --xsecs $xsecs_path
	    continue
	else
	    for region in "${regions[@]}"; do
       		python process_histograms.py --hpath $hists_path --sample $sample --histogram $hist --region $region --lumi $lumi --xsecs $xsecs_path
	    done
	fi
    done
done
