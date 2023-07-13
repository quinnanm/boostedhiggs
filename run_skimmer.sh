#!/usr/bin/bash

# run over signal files
mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0 --label H --inference
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/train

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 1 --label H --inference
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/test

# # run over TTbar files
# mkdir -p ntuples/TTToSemiLeptonic/

# python run.py --processor input --local --sample TTToSemiLeptonic --n 50 --starti 0 --label Top --inference
# mv outfiles ntuples/TTToSemiLeptonic/train

# python run.py --processor input --local --sample TTToSemiLeptonic --n 50 --starti 1 --label Top --inference
# mv outfiles ntuples/TTToSemiLeptonic/test

# # run over WJetsLNu files
# for SAMPLE in WJetsToLNu_HT-200To400 WJetsToLNu_HT-400To600 WJetsToLNu_HT-600To800 WJetsToLNu_HT-800To1200
# do
#     mkdir -p ntuples/$SAMPLE/

#     python run.py --processor input --local --sample $SAMPLE --n 5 --starti 0 --label VJets --inference
#     mv outfiles ntuples/$SAMPLE/train

#     python run.py --processor input --local --sample $SAMPLE --n 5 --starti 1 --label VJets --inference
#     mv outfiles ntuples/$SAMPLE/test
# done

# # run over QCD files
# for SAMPLE in QCD_Pt_300to470 QCD_Pt_470to600 QCD_Pt_600to800 QCD_Pt_800to1000
# do
#     mkdir -p ntuples/$SAMPLE/

#     python run.py --processor input --local --sample $SAMPLE --n 5 --starti 0 --label QCD --inference
#     mv outfiles ntuples/$SAMPLE/train

#     python run.py --processor input --local --sample $SAMPLE --n 5 --starti 1 --label QCD --inference
#     mv outfiles ntuples/$SAMPLE/test
# done

# -----------------------------------------------------------------------------
# MERGING ROOTFILES IN A CMSENV

# merging signal files
cd ntuples/GluGluHToWW_Pt-200ToInf_M-125/train
hadd -fk out.root outroot/*/*
rm -r outroot
cd ../test
hadd -fk out.root outroot/*/*
rm -r outroot
cd ../../

# # merging TTbar files
# cd ntuples/TTToSemiLeptonic/train
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../test
# hadd -fk out.root outroot/*/*
# rm -r outroot
# cd ../../

# # merging WJetsLNu files
# for SAMPLE in WJetsToLNu_HT-200To400 WJetsToLNu_HT-400To600 WJetsToLNu_HT-600To800 WJetsToLNu_HT-800To1200
# do
#     cd ntuples/$SAMPLE/train
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../test
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../../
# done

# # merging QCD files
# for SAMPLE in QCD_Pt_300to470 QCD_Pt_470to600 QCD_Pt_600to800 QCD_Pt_800to1000
# do
#     cd ntuples/$SAMPLE/train
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../test
#     hadd -fk out.root outroot/*/*
#     rm -r outroot
#     cd ../../
# done
