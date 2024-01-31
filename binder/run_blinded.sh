#!/bin/bash

### https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/combine/run_blinded.sh

####################################################################################################
# Script for fits
#
# 1) Combines cards and makes a workspace (--workspace / -w)
# 2) Background-only fit (--bfit / -b)
# 3) Expected asymptotic limits (--limits / -l)
# 4) Expected significance (--significance / -s)
# 5) Fit diagnostics (--dfit / -d)
# 6) GoF on data (--gofdata / -g)
# 7) GoF on toys (--goftoys / -t),
# 8) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impactsf $nuisance), collect (--impactsc $nuisances)
# 9) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
#
# Specify seed with --seed (default 42) and number of toys with --numtoys (default 100)
#
# Usage ./run_blinded.sh [-wblsdgt] [--numtoys 100] [--seed 42]
####################################################################################################


####################################################################################################
# Read options
####################################################################################################

workspace=0
bfit=0
limits=0
significance=0
dfit=0
gofdata=0
goftoys=0
impactsi=0
impactsf=0
impactsc=0
seed=444
numtoys=100
bias=-1
mintol=0.5 # --cminDefaultMinimizerTolerance
# maxcalls=1000000000  # --X-rtd MINIMIZER_MaxCalls

options=$(getopt -o "wblsdrgti" --long "workspace,bfit,limits,significance,dfit,gofdata,goftoys,impactsi,impactsf:,impactsc:,bias:,seed:,numtoys:,mintol:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        -s|--significance)
            significance=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        -i|--impactsi)
            impactsi=1
            ;;
        --impactsf)
            shift
            impactsf=$1
            ;;
        --impactsc)
            shift
            impactsc=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --mintol)
            shift
            mintol=$1
            ;;
        --bias)
            shift
            bias=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

echo "Arguments: workspace=$workspace bfit=$bfit limits=$limits \
significance=$significance dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys"



####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is turned off)
####################################################################################################

dataset=data_obs
cards_dir="/uscms/home/fmokhtar/nobackup/boostedhiggs/combine/templates/v18/datacards"
cp ${cards_dir}/testModel.root testModel.root # TODO: avoid this
CMS_PARAMS_LABEL="CMS_HWW_boosted"

# ####################################################################################################
# # Combine cards, text2workspace, fit, limits, significances, fitdiagnositcs, GoFs
# ####################################################################################################
# # # need to run this for large # of nuisances
# # # https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735


# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

logsdir=${outdir}/logs
mkdir -p $logsdir
chmod +x ${logsdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

sr1B="SR1VBFBlinded"
sr2B="SR1ggFpt300to450Blinded"
sr3B="SR1ggFpt450toInfBlinded"
sr4B="SR2Blinded"

cr1B="WJetsCRfor${sr1B}"
cr2B="WJetsCRfor${sr2B}"
cr3B="WJetsCRfor${sr3B}"
cr4B="WJetsCRfor${sr4B}"

# ccargs="${sr1B}=${cards_dir}/${sr1B}.txt ${cr1B}=${cards_dir}/${cr1B}.txt"
# ccargs="${sr2B}=${cards_dir}/${sr2B}.txt ${cr2B}=${cards_dir}/${cr2B}.txt"
# ccargs="${sr3B}=${cards_dir}/${sr3B}.txt ${cr3B}=${cards_dir}/${cr3B}.txt"
# ccargs="${sr2B}=${cards_dir}/${sr2B}.txt"
# ccargs="SR=${cards_dir}/${sr3B}.txt"

ccargs="SR1=${cards_dir}/${sr1B}.txt SR2=${cards_dir}/${sr2B}.txt SR3=${cards_dir}/${sr3B}.txt SR4=${cards_dir}/${sr4B}.txt"
ccargs+=" CR1=${cards_dir}/${cr1B}.txt CR2=${cards_dir}/${cr2B}.txt CR3=${cards_dir}/${cr3B}.txt CR4=${cards_dir}/${cr4B}.txt"


ccargs="SR1=${cards_dir}/${sr1B}.txt"
ccargs+=" CR1=${cards_dir}/${cr1B}.txt"


# ccargs="SR2=${cards_dir}/${sr2B}.txt"
# ccargs="SR1=${cards_dir}/${sr1B}.txt SR2=${cards_dir}/${sr2B}.txt"

# params WITH SAME NAME across channels in combine cards are tied up
# maskunblindedargs="mask_${sr1}=1"
# maskblindedargs="mask_${sr1B}=1"

# maskblindedargs=${maskblindedargs%,}
# maskunblindedargs=${maskunblindedargs%,}
# echo "cards args=${ccargs}"

# freeze qcd params in blinded bins
setparamsblinded=""
freezeparamsblinded=""
for bin in {2..5}
do
    setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin}=0,"
    freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin},"
done
# remove last comma
setparamsblinded=${setparamsblinded%,}
freezeparamsblinded="${freezeparamsblinded}"


if [ $workspace = 1 ]; then
    echo "Combining cards"
    echo $ccargs
    combineCards.py $ccargs > $combined_datacard

    echo "Running text2workspace"
    text2workspace.py $combined_datacard --channel-masks -o $ws 2>&1 | tee $logsdir/text2workspace.txt
else
    if [ ! -f "$ws" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi

# if [ $dfit = 1 ]; then
#     echo "Fit Diagnostics"
#     combine -M FitDiagnostics -m 125 -d $ws \
#     --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance $mintol --X-rtd MINIMIZER_MaxCalls=5000000 \
#     -n Blinded --ignoreCovWarning -v 13 --robustFit=1 \
#     --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

#     # combine -M FitDiagnostics -m 125 -d $ws \
#     # --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance $mintol --X-rtd MINIMIZER_MaxCalls=5000000 \
#     # -n Blinded --ignoreCovWarning -v 13 --minos none\
#     # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt


#     # combine -M FitDiagnostics -m 125 -d $ws \
#     # --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance $mintol --X-rtd MINIMIZER_MaxCalls=5000000 \
#     # -n Blinded --ignoreCovWarning -v 13 \
#     # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt
# fi

if [ $dfit = 1 ]; then
    echo "Fit Diagnostics"

    # combine -M FitDiagnostics -d $ws -t -1 --expectSignal 1 \
    # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

    # combine -M FitDiagnostics -d $ws --expectSignal 1 \
    # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

    combine -M FitDiagnostics -d $ws --expectSignal 1 \
    --setParameters ${setparamsblinded} \
    --freezeParameters ${freezeparamsblinded} \
    --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

    # do: --skipSBFit
    # echo "Fit Diagnostics"
    # combine -M FitDiagnostics -m 125 -d $ws \
    # --setParameters ${maskunblindedargs},${setparamsblinded} \
    # --freezeParameters ${freezeparamsblinded} \
    # --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance $mintol --X-rtd MINIMIZER_MaxCalls=5000000 \
    # -n Blinded --ignoreCovWarning -v 13 \
    # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes 2>&1 | tee $logsdir/FitDiagnostics.txt

fi


if [ $significance = 1 ]; then
    echo "Expected significance"

    ### EXPECTED SIGNIFICANCE: must use unblinded templates with asimov dataset (-t -1)
    xxx=${cards_dir}/combined_${category}/workspace
    combine -M Significance ${xxx}.root -m 125 --rMin -1 --rMax 5 -t -1 --expectSignal 1

    ### EXPECTED SIGNIFICANCE: must use unblinded templates with asimov dataset (-t -1)
    # wsm_snapshot=higgsCombineBlinded.FitDiagnostics.mH125

    # combine -M Significance -d ${wsm_snapshot}.root -n "" --significance -m 125 --snapshotName MultiDimFit -v 9 \
    # -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
    # ${unblindedparams},r=1 \
    # --floatParameters ${freezeparamsblinded},r --toysFrequentist 2>&1 | tee $outsdir/Significance.txt

    # combine -M Significance -d ${wsm_snapshot}.root --snapshotName MultiDimFit -m 125 -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit

fi


# if [ $bfit = 1 ]; then
#     echo "Blinded background-only fit"
#     combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 \
#     --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance $mintol --X-rtd MINIMIZER_MaxCalls=5000000 \
#     --setParameters ${maskunblindedargs},${setparamsblinded},r=0  \
#     --freezeParameters r,${freezeparamsblinded} \
#     -n Snapshot 2>&1 | tee $logsdir/MultiDimFit.txt
# else
#     if [ ! -f "higgsCombineSnapshot.MultiDimFit.mH125.root" ]; then
#         echo "Background-only fit snapshot doesn't exist! Use the -b|--bfit option to run fit first"
#         exit 1
#     fi
# fi

# # wsm_snapshot=/uscms/home/fmokhtar/nobackup/boostedhiggs/combine/templates/v2/datacards/testModel
# if [ $limits = 1 ]; then
#     echo "Expected limits"
#     combine -M AsymptoticLimits -m 125 -n "" -d ${wsm}.root --snapshotName MultiDimFit -v 9 \
#     --saveWorkspace --saveToys --bypassFrequentistFit \
#     ${unblindedparams},r=0 -s $seed \
#     --floatParameters ${freezeparamsblinded},r --toysFrequentist --run blind 2>&1 | tee $logsdir/AsymptoticLimits.txt
# fi



# # try to change "setparams" to "setparamsblinded" and see the effect
# # if [ $gofdata = 1 ]; then
# #     echo "GoF on data"
# #     combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
# #     --snapshotName MultiDimFit --bypassFrequentistFit \
# #     --setParameters ${maskunblindedargs},${setparams},r=0 \
# #     --freezeParameters ${freezeparams},r \
# #     -n Data -v 9 2>&1 | tee $logsdir/GoF_data.txt
# # fi

# if [ $gofdata = 1 ]; then
#     echo "GoF on data"
#     combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
#     --snapshotName MultiDimFit --bypassFrequentistFit \
#     --setParameters ${maskunblindedargs},${setparams},r=0 \
#     --freezeParameters ${freezeparams},r \
#     -n Data -v 9 2>&1 | tee $logsdir/GoF_data.txt
# fi


# if [ $goftoys = 1 ]; then
#     # echo ${freezeparams} "test value"
#     echo "GoF on toys" #always bug.
#     echo ${maskunblindedargs}
#     # combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
#     # --snapshotName MultiDimFit --bypassFrequentistFit \
#     # --setParameters ${maskunblindedargs},${setparams},r=0 \
#     # --freezeParameters ${freezeparamsblinded} --saveToys \
#     # -n Toys   -s $seed -t $numtoys --toysFrequentist 2>&1 | tee $logsdir/GoF_toys.txt

#     echo "GoF on toys"
#     combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
#     --snapshotName MultiDimFit --bypassFrequentistFit \
#     --setParameters ${maskunblindedargs},r=0 \
#     --freezeParameters r --saveToys \
#     -n Toys  -v 9 -s $seed -t $numtoys --toysFrequentist 2>&1 | tee $logsdir/GoF_toys.txt

# fi


# if [ $impactsi = 1 ]; then
#     echo "Initial fit for impacts"
#     # from https://github.com/cms-analysis/CombineHarvester/blob/f0e0c53298521921abf59c175b5c5616026d203b/CombineTools/python/combine/Impacts.py#L113
#     # combine -M MultiDimFit -m 125 -n "_initialFit_impacts" -d ${wsm_snapshot}.root --snapshotName MultiDimFit \
#     #  --algo singles --redefineSignalPOIs r --floatOtherPOIs 1 --saveInactivePOI 1 -P r --setParameterRanges r=-0.5,20 \
#     # --toysFrequentist --expectSignal 1 --bypassFrequentistFit -t -1 \
#     # ${unblindedparams} --floatParameters ${freezeparamsblinded} \
#     # --robustFit 1 --cminDefaultMinimizerStrategy=1 -v 9 2>&1 | tee $logsdir/Impacts_init.txt

#     combineTool.py -M Impacts --snapshotName MultiDimFit -m 125 -n "impacts" \
#     -t -1 --bypassFrequentistFit --toysFrequentist --expectSignal 1 \
#     -d ${wsm_snapshot}.root --doInitialFit --robustFit 1 \
#     ${unblindedparams} --floatParameters ${freezeparamsblinded} \
#      --cminDefaultMinimizerStrategy=0 -v 1 2>&1 | tee $logsdir/Impacts_init.txt

#     # plotImpacts.py -i impacts.json -o impacts
# fi


# if [ $impactsf != 0 ]; then
#     echo "Submitting jobs for impact scans"
#     # Impacts module cannot access parameters which were frozen in MultiDimFit, so running impacts
#     # for each parameter directly using its internal command
#     # (also need to do this for submitting to condor anywhere other than lxplus)
#     combine -M MultiDimFit -n _paramFit_impacts_$impactsf --algo impact --redefineSignalPOIs r -P $impactsf \
#     --floatOtherPOIs 1 --saveInactivePOI 1 --snapshotName MultiDimFit -d ${wsm_snapshot}.root \
#     -t -1 --bypassFrequentistFit --toysFrequentist --expectSignal 1 --robustFit 1 \
#     ${unblindedparams} --floatParameters ${freezeparamsblinded} \
#     --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 1 -m 125 | tee $logsdir/Impacts_$impactsf.txt

#     # Old Impacts command:
#     # combineTool.py -M Impacts -t -1 --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal 1 \
#     # -m 125 -n "impacts" -d ${wsm_snapshot}.root --doFits --robustFit 1 \
#     # --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} \
#     # --exclude ${excludeimpactparams} \
#     # --job-mode condor --dry-run \
#     # --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 9 2>&1 | tee $logsdir/Impacts_fits.txt
# fi


# if [ $impactsc != 0 ]; then

#     echo "Collecting impacts"

#     # combineTool.py -M Impacts --snapshotName MultiDimFit \
#     # -m 125 -n "impacts" -d ${wsm_snapshot}.root \
#     # --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} \
#     # -t -1 --named $impactsc \
#     # --setParameterRanges r=-0.5,20 -v 1 -o impacts.json 2>&1 | tee $logsdir/Impacts_collect.txt

#     # plotImpacts.py -i impacts.json -o impacts

#     combineTool.py -M Impacts -d ${wsm_snapshot}.root --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} --cminDefaultMinimizerStrategy=0 --expectSignal=1 -t -1 -m 125 --doInitialFit --robustFit 1   --rMin -40 --rMax 40
#     combineTool.py -M Impacts -d ${wsm_snapshot}.root --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} --cminDefaultMinimizerStrategy=0 --expectSignal=1 -t -1 -m 125 --robustFit 1 --doFits  --rMin -40 --rMax 40
#     combineTool.py -M Impacts -d ${wsm_snapshot}.root --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} --cminDefaultMinimizerStrategy=0 --expectSignal=1 -t -1 -m 125 -o impacts.json  --rMin -40 --rMax 40

#     plotImpacts.py -i impacts.json -o impacts
# fi


# if [ $bias != -1 ]; then
#     echo "Bias test with bias $bias"
#     # setting verbose > 0 here can lead to crazy large output files (~10-100GB!) because of getting
#     # stuck in negative yield areas
#     combine -M FitDiagnostics --trackParameters r --trackErrors r --justFit \
#     -m 125 -n "bias${bias}" -d ${wsm_snapshot}.root --rMin "-15" --rMax 15 \
#     --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal $bias \
#     ${unblindedparams},r=$bias --floatParameters ${freezeparamsblinded} \
#     --robustFit=1 -t $numtoys -s $seed \
#     --X-rtd MINIMIZER_MaxCalls=1000000 --cminDefaultMinimizerTolerance $mintol 2>&1 | tee $logsdir/bias${bias}seed${seed}.txt
# fi
