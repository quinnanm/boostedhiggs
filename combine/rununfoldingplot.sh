#!/bin/bash

#run like
#./rununfoldingplot.sh --cardsdir "templates/v7/datacards_unfolding"


# Default value for cards_dir
cards_dir="templates/v17/datacards_unfolding"

# Parse the --cardsdir argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cardsdir) cards_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Now the value of cards_dir is either the default or provided by the user
# cards_dir="templates/v7/datacards"
# cards_dir="templates/v7/datacards_unfolding"

dataset=data_obs
cp ${cards_dir}/testModel.root testModel.root # TODO: avoid this
CMS_PARAMS_LABEL="CMS_HWW_boosted"

# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

logsdir=${outdir}/logs
mkdir -p $logsdir
chmod +x ${logsdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

# # ADD REGIONS
sr1="VBF"
sr2="ggFpt250to350"
sr3="ggFpt350to500"
sr4="ggFpt500toInf"
ccargs="SR1=${cards_dir}/${sr1}.txt SR2=${cards_dir}/${sr2}.txt SR3=${cards_dir}/${sr3}.txt SR4=${cards_dir}/${sr4}.txt"

cr1="TopCR"
cr2="WJetsCR"
ccargs+=" CR1=${cards_dir}/${cr1}.txt CR2=${cards_dir}/${cr2}.txt"

echo "Combining cards:"
for file in $ccargs; do
    echo "  ${file##*/}"
done
echo "-------------------------"
combineCards.py $ccargs > $combined_datacard

echo "Running text2workspace"

text2workspace.py $combined_datacard -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ggH_hww_200_300:r_ggH_pt200_300[1,-10,10]' --PO 'map=.*/ggH_hww_300_450:r_ggH_pt300_450[1,-10,10]' --PO 'map=.*/ggH_hww_450_Inf:r_ggH_pt450_inf[1,-10,10]' --PO 'map=.*/qqH_hww_mjj_1000_Inf:r_qqH_hww_mjj_1000_Inf[1,-10,10]' --PO 'map=.*/WH_hww:r_WH_hww[1,-10,10]' --PO 'map=.*/ZH_hww:r_ZH_hww[1,-10,10]' --PO 'map=.*/ttH_hww:r_ttH_hww[1,-10,10]' -o $ws 2>&1 | tee $logsdir/text2workspace.txt

echo "Running multidimfit"

#combineTool.py -M FitDiagnostics -d $ws --saveWorkspace

#ggf + vbf floating others fixed
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 

# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww 

# echo "vbf floating, fix ggFpt250to350, ggFpt350to500, ggFpt500toInf"
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_ggH_pt200_300,r_ggH_pt300_450,r_ggH_pt450_inf,r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH

# echo "ggFpt250to350 floating, fix vbf, ggFpt350to500, ggFpt500toInf"
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_qqH_hww,r_ggH_pt300_450,r_ggH_pt450_inf,r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH

# echo "ggFpt350to500 floating, fix vbf, ggFpt250to350, ggFpt500toInf"
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_ggH_pt200_300,r_qqH_hww,r_ggH_pt450_inf,r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH

# echo "ggFpt500toInf floating, fix vbf, ggFpt250to350, ggFpt350to500"
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_ggH_pt200_300,r_ggH_pt300_450,r_qqH_hww,r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH


#standard unfolding original:
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww_mjj_1000_Inf=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH,pdf_Higgs_ZH,pdf_Higgs_qqH,pdf_Higgs_ttH,alpha_s,QCDscale_WH,QCDscale_WH_ACCEPT_CMS_HWW_boosted,QCDscale_ZH,QCDscale_ZH_ACCEPT_CMS_HWW_boosted,QCDscale_qqH,QCDscale_qqH_ACCEPT_CMS_HWW_boosted,QCDscale_singletop_ACCEPT_CMS_HWW_boosted,QCDscale_ttH,QCDscale_ttH_ACCEPT_CMS_HWW_boosted,QCDscale_wjets_ACCEPT_CMS_HWW_boosted>multidimresults.txt

#unfolding updated uncertainties
# combine -M MultiDimFit --algo singles -d $ws -t -1 --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww_mjj_1000_Inf=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww,pdf_Higgs_WH,pdf_Higgs_ZH,pdf_Higgs_ttH,ps_fsr_WH,ps_fsr_ZH,ps_fsr_singletop,ps_fsr_ttH,ps_fsr_ttbar,ps_fsr_wjets,ps_isr_WH,ps_isr_ZH,ps_isr_singletop,ps_isr_ttH,ps_isr_ttbar,ps_isr_wjets,PDF_WH_ACCEPT_CMS_HWW_boosted,PDF_ZH_ACCEPT_CMS_HWW_boosted,PDF_ttH_ACCEPT_CMS_HWW_boosted,PDF_ttbar_ACCEPT_CMS_HWW_boosted,PDF_wjets_ACCEPT_CMS_HWW_boosted,QCDscale_WH,QCDscale_WH_ACCEPT_CMS_HWW_boosted,QCDscale_ZH,QCDscale_ZH_ACCEPT_CMS_HWW_boosted,QCDscale_singletop_ACCEPT_CMS_HWW_boosted,QCDscale_ttH,QCDscale_ttH_ACCEPT_CMS_HWW_boosted,QCDscale_wjets_ACCEPT_CMS_HWW_boosted,alpha_s>multidimresults.txt

#unblinded v13 
# combine -M MultiDimFit --algo singles -d $ws --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww_mjj_1000_Inf=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww,alpha_s,QCDscale_qqH_mjj1000toInf,QCDscale_ggH_pt200to300,QCDscale_ggH_pt300to450,QCDscale_ggH_pt450toInf,pdf_Higgs_qqH_mjj1000toInf,pdf_Higgs_ggH_pt450toInf,pdf_Higgs_ggH_pt300to450,pdf_Higgs_ggH_pt200to300,ps_isr_ggH_pt200to300,ps_isr_ggH_pt300to450,ps_isr_ggH_pt450toInf,ps_isr_qqH_mjj1000toInf,ps_fsr_ggH_pt200to300,ps_fsr_ggH_pt300to450,ps_fsr_ggH_pt450toInf,ps_fsr_qqH_mjj1000toInf>multidimresults.txt

#unblinded v17
combine -M MultiDimFit --algo singles -d $ws --setParameters r_ggH_pt200_300=1,r_ggH_pt300_450=1,r_ggH_pt450_inf=1,r_qqH_hww_mjj_1000_Inf=1,r_WH_hww=1,r_ZH_hww=1,r_ttH_hww=1 --freezeParameters r_WH_hww,r_ZH_hww,r_ttH_hww,QCDscale_ggH_ACCEPT_CMS_HWW_boosted,QCDscale_qqH_ACCEPT_CMS_HWW_boosted,PDF_ggH_ACCEPT_CMS_HWW_boosted,PDF_qqH_ACCEPT_CMS_HWW_boosted,ps_isr_qqH,ps_isr_ggH,ps_fsr_qqH,ps_fsr_ggH,alpha_s>multidimresults.txt

echo "Multidimfit results saved to multidimresults.txt"

echo "cards dir:"
echo ${cards_dir}
