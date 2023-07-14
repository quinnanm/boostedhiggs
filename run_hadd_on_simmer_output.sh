#!/usr/bin/bash

# MERGING ROOTFILES IN A CMSENV

# # merging signal files
# cd ntuples/GluGluHToWW_Pt-200ToInf_M-125/train
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../test
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../../../

# # merging TTbar files
# cd ntuples/TTToSemiLeptonic/train
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../test
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../../../

# # merging WJetsLNu files
# for SAMPLE in WJetsToLNu_HT-200To400 WJetsToLNu_HT-400To600 WJetsToLNu_HT-600To800 WJetsToLNu_HT-800To1200
# do
#     cd ntuples/$SAMPLE/train
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../test
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../../../
# done

# merging QCD files
for SAMPLE in QCD_Pt_300to470 QCD_Pt_470to600 QCD_Pt_600to800 QCD_Pt_800to1000
do
    cd ntuples/$SAMPLE/train
    hadd -fk out.root outroot/*/*
    rm -r outroot
    cd ../test
    hadd -fk out.root outroot/*/*
    rm -r outroot
    cd ../../../
done
