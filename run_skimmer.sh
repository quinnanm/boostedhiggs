#!/usr/bin/bash

# run over signal files
mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 10000 --starti 0 --label H --inference --year 2018
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2018

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 10000 --starti 0 --label H --inference --year 2017
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2017

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 10000 --starti 0 --label H --inference --year 2016APV
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016APV

python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 10000 --starti 0 --label H --inference --year 2016
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016
