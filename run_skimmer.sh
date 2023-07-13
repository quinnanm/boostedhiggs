#!/usr/bin/bash

# run over signal files
mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 0 --label H --inference
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/train

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 1 --starti 1 --label H --inference
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/test
