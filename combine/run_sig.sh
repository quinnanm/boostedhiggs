dataset=data_obs
cards_dir="/uscms/home/fmokhtar/nobackup/boostedhiggs/combine/templates/v39/datacards"
cp ${cards_dir}/testModel.root testModel.root # TODO: avoid this
CMS_PARAMS_LABEL="CMS_HWW_boosted"

# ####################################################################################################
# # Combine cards, significances
# ####################################################################################################

# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

cat0=VBF99
ccargs="${cat0}=${cards_dir}/${cat0}.txt"

cat1=ggF985
cat2=VBF97
cat3=SR2ggF96and985

cat0=VBF97
cat1=SR1ggFpt250to300
cat2=SR1ggFpt300to450
cat3=SR1ggFpt450toInf
cat4=SR2ggF

# ccargs="${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt"
# ccargs="${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt"

ccargs="${cat0}=${cards_dir}/${cat0}.txt ${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt ${cat4}=${cards_dir}/${cat4}.txt"
# ccargs="${cat0}=${cards_dir}/${cat0}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt ${cat4}=${cards_dir}/${cat4}.txt"

# ccargs="${cat2}=${cards_dir}/${cat2}.txt"

# ccargs="${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt"
# ccargs="${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt ${cat10}=${cards_dir}/${cat10}.txt"
# ccargs+=" ${cr}=${cards_dir}/${cr}.txt"

# ccargs="${cat0}=${cards_dir}/${cat0}.txt ${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt"

# ccargs="${cat0}=${cards_dir}/${cat0}.txt ${cat1}=${cards_dir}/${cat1}.txt ${cat2}=${cards_dir}/${cat2}.txt ${cat3}=${cards_dir}/${cat3}.txt ${cat10}=${cards_dir}/${cat10}.txt"


echo "Combining cards:"
for file in $ccargs; do
  echo "  ${file##*/}"
done
echo "-------------------------"
combineCards.py $ccargs > $combined_datacard

echo "Running text2workspace"
text2workspace.py $combined_datacard --channel-masks -o $ws 2>&1

echo "Expected significance"
combine -M Significance -d $ws -m 125 -t -1 --expectSignal=1 --rMin -1 --rMax 5
