#!/bin/bash

# run as: bash plot.sh $DATE

# histograms directory                                                                                                                          
indir="/uscms/home/docampoh/nobackup/boostedhiggs/python/hists"

# output directory                                                                                                                              
output="./plots/$1"

# histograms
hists=(jet_kin met_kin lep_kin cutflow jet_lep)

# regions                                                                                                                                       
channel=(hadel hadmu)

# axis                                                                                                                                     
jet_kin=(jetpt jetmsd jetrho btag)
met_kin=(met mt_lepmet)
lep_kin=(lepminiIso leprelIso lep_pt deltaR_lepjet)
jet_lep=(jetlep_msd jetlep_mass)

for region in "${channel[@]}"; do
    for hist in "${hists[@]}"; do
	if [[ $hist == "jet_kin" ]]; then
	    for var in "${jet_kin[@]}"; do
		python plot_stack.py --hname $hist --haxis $var --year 2017 --channel $region --idir $indir --odir $output
            done
        fi
        if [[ $hist == "met_kin" ]]; then
	    for var in "${met_kin[@]}"; do
		python plot_stack.py --hname $hist --haxis $var --year 2017 --channel $region --idir $indir --odir $output
	    done
	fi
	if [[ $hist == "lep_kin" ]]; then
	    for var in "${lep_kin[@]}"; do
		python plot_stack.py --hname $hist --haxis $var --year 2017 --channel $region --idir $indir --odir $output
	    done
        fi
        if [[ $hist == "cutflow" ]]; then
	    python plot_stack.py --hname $hist --haxis cut --year 2017 --channel $region --idir $indir --odir $output
        fi
	if [[ $hist == "jet_lep" ]]; then
	    for var in "${jet_lep[@]}"; do
		python plot_stack.py --hname $hist --haxis $var --year 2017 --channel $region --idir $indir --odir $output
	    done
	fi
   done
done
