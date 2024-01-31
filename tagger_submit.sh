#!/usr/bin/bash

###############################################################################################################
# The following run.py commands on the signal samples are chosen following roughly a 60-40 train-test split
# after considering the number of files available. For reference, the number of files available are,
    # ggF number of files:
    # - 2018: 22
    # - 2017: 25
    # - 2016APV: 22
    # - 2016: 15
    # VBF number of files:
    # - 2018: 7
    # - 2017: 18
    # - 2016APV: 7
    # - 2016: 16
###############################################################################################################

python condor/submit.py --channels ele,mu --year 2018 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_sig --pfnano v2_2 --processor input --inference --submit
python condor/submit.py --channels ele,mu --year 2017 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_sig --pfnano v2_2 --processor input --inference --submit
python condor/submit.py --channels ele,mu --year 2016 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_sig --pfnano v2_2 --processor input --inference --submit
python condor/submit.py --channels ele,mu --year 2016APV --tag TaggerInput --config samples_inclusive.yaml --key finetuning_sig --pfnano v2_2 --processor input --inference --submit

python condor/submit.py --channels ele,mu --year 2018 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_top --pfnano v2_2 --processor input --inference --submit --maxfiles 500
python condor/submit.py --channels ele,mu --year 2018 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_wjets --pfnano v2_2 --processor input --inference --submit --maxfiles 100
python condor/submit.py --channels ele,mu --year 2018 --tag TaggerInput --config samples_inclusive.yaml --key finetuning_qcd --pfnano v2_2 --processor input --inference --submit --maxfiles 50
