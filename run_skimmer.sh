#!/usr/bin/bash

# run over signal samples (ggF)
mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/2018
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 13 --starti 0 --inference --year 2018
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2018/train
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 13 --starti 1 --inference --year 2018
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2018/test

mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/2017
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 15 --starti 0 --inference --year 2017
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2017/train
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 15 --starti 1 --inference --year 2017
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2017/test

mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016APV
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 13 --starti 0 --inference --year 2016APV
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016APV/train
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 13 --starti 1 --inference --year 2016APV
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016APV/test

mkdir -p ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 9 --starti 0 --inference --year 2016
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016/train
python run.py --processor input --local --sample GluGluHToWW_Pt-200ToInf_M-125 --n 9 --starti 1 --inference --year 2016
mv outfiles ntuples/GluGluHToWW_Pt-200ToInf_M-125/2016/test

# run over signal samples (VBF)
mkdir -p ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2018
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 4 --starti 0 --inference --year 2018
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2018/train
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 4 --starti 1 --inference --year 2018
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2018/test

mkdir -p ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2017
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 11 --starti 0 --inference --year 2017
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2017/train
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 11 --starti 1 --inference --year 2017
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2017/test

mkdir -p ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016APV
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 4 --starti 0 --inference --year 2016APV
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016APV/train
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 4 --starti 1 --inference --year 2016APV
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016APV/test

mkdir -p ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 9 --starti 0 --inference --year 2016
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016/train
python run.py --processor input --local --sample VBFHToWWToLNuQQ_M-125_withDipoleRecoil --n 9 --starti 1 --inference --year 2016
mv outfiles ntuples/VBFHToWWToLNuQQ_M-125_withDipoleRecoil/2016/test

# run over Top samples
mkdir -p ntuples/TTToSemiLeptonic/2018/
python run.py --processor input --local --sample TTToSemiLeptonic --n 10 --starti 0 --inference --year 2018
mv outfiles ntuples/TTToSemiLeptonic/2018/train
python run.py --processor input --local --sample TTToSemiLeptonic --n 10 --starti 1 --inference --year 2018
mv outfiles ntuples/TTToSemiLeptonic/2018/test

# run over WJetsLNu samples
for SAMPLE in WJetsToLNu_HT-200To400 WJetsToLNu_HT-400To600 WJetsToLNu_HT-600To800 WJetsToLNu_HT-800To1200
do
    mkdir -p ntuples/$SAMPLE/2018/

    python run.py --processor input --local --sample $SAMPLE --n 5 --starti 0 --inference --year 2018
    mv outfiles ntuples/$SAMPLE/2018/train

    python run.py --processor input --local --sample $SAMPLE --n 5 --starti 1 --inference --year 2018
    mv outfiles ntuples/$SAMPLE/2018/test
done

# run over QCD files
for SAMPLE in QCD_Pt_170to300 QCD_Pt_300to470 QCD_Pt_470to600 QCD_Pt_600to800
do
    mkdir -p ntuples/$SAMPLE/2018/

    python run.py --processor input --local --sample $SAMPLE --n 5 --starti 0 --inference --year 2018
    mv outfiles ntuples/$SAMPLE/2018/train

    python run.py --processor input --local --sample $SAMPLE --n 5 --starti 1 --inference --year 2018
    mv outfiles ntuples/$SAMPLE/2018/test
done
