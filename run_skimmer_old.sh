#!/usr/bin/bash

# # run over signal files
# mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/

# python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 18 --starti 0 --label H --inference
# mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/train

# python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 18 --starti 1 --label H --inference
# mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/test


# run over variable mass higgs files
mkdir -p ntuples/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/

python run.py --processor input --local --sample BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2 --n 125 --starti 0 --label H --inference
mv outfiles ntuples/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/train

python run.py --processor input --local --sample BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2 --n 125 --starti 1 --label H --inference
mv outfiles ntuples/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/test

# # run over TTbar files
# mkdir -p ntuples/TTToSemiLeptonic/

python run.py --processor input --local --sample TTToSemiLeptonic --n 50 --starti 3 --label Top --inference --year 2018
mv outfiles ntuples/TTToSemiLeptonic/train

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
