#!/bin/bash

# run as: bash run.sh $DATE


# Loading histograms

# path to output
hists_path="/eos/uscms/store/user/cmantill/boostedhiggs/Sep14_nopuw_UL/outfiles/"

# path to cross section json 
xsecs_path="/uscms/home/docampoh/nobackup/boostedhiggs/fileset/xsecs.json"

# samples to load
samples=(hww tt_semileptonic tt_hadronic tt_dileptonic qcd wjets st zjets electron muon)

# histograms to load
hists=(jet_kin met_kin cutflow)

# luminosity
lumi=41500


for sample in "${samples[@]}"
do
    for hist in "${hists[@]}"
    do
	python process_histograms.py  --hpath $hists_path --sample $sample --histogram $hist --lumi $lumi --xsecs $xsecs_path
    done
done


# Plotting

# histograms directory
indir="/uscms/home/docampoh/nobackup/boostedhiggs/python/hists"

# output directory
output="./plots/$1"

# regions
channel=(hadel hadmu)

# jet axis 
jet=(jetpt jetmsd jetrho btag)


for region in "${channel[@]}"
do
    for hist in "${hists[@]}"
    do 
	if [[ $hist == "jet_kin" ]]
	then
	    for var in "${jet[@]}"
	    do
		python plot_stack.py --hname $hist --haxis $var --year 2017 --channel $region --idir $indir --odir $output
	    done
	fi
	if [[ $hist == "met_kin" ]]
	then
	    python plot_stack.py --hname $hist --haxis met --year 2017 --channel $region --idir $indir --odir $output
	fi
	if [[ $hist == "cutflow" ]]
	then
	    python plot_stack.py --hname $hist --haxis cut --year 2017 --channel $region --idir $indir --odir $output
	fi
   done
done
