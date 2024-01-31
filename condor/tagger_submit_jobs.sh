#!/bin/bash

python condor/tagger_submit.py --year 2018 --tag train --sample TTToSemiLeptonic --n 100 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample TTToSemiLeptonic --n 100 --starti 1 --submit

python condor/tagger_submit.py --year 2018 --tag train --sample WJetsToLNu_HT-200To400 --n 10 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag train --sample WJetsToLNu_HT-400To600 --n 10 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag train --sample WJetsToLNu_HT-600To800 --n 10 --starti 0 --submit

python condor/tagger_submit.py --year 2018 --tag test --sample WJetsToLNu_HT-200To400 --n 10 --starti 1 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample WJetsToLNu_HT-400To600 --n 10 --starti 1 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample WJetsToLNu_HT-600To800 --n 10 --starti 1 --submit

python condor/tagger_submit.py --year 2018 --tag train --sample QCD_Pt_170to300 --n 5 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag train --sample QCD_Pt_300to470 --n 5 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag train --sample QCD_Pt_470to600 --n 5 --starti 0 --submit
python condor/tagger_submit.py --year 2018 --tag train --sample QCD_Pt_600to800 --n 5 --starti 0 --submit

python condor/tagger_submit.py --year 2018 --tag test --sample QCD_Pt_170to300 --n 5 --starti 1 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample QCD_Pt_300to470 --n 5 --starti 1 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample QCD_Pt_470to600 --n 5 --starti 1 --submit
python condor/tagger_submit.py --year 2018 --tag test --sample QCD_Pt_600to800 --n 5 --starti 1 --submit
