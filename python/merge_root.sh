#!/bin/bash

# run after running convert_to_root.py to merge the rootfiles under merged/2017
IDIR="rootfiles"
ODIR="merged"

set -e

mkdir -p ${ODIR}/2017

# process the cms data
for ch in ${IDIR}/* ; do
  channel=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$ch") # get channel as a variable
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")  # get sample as a variable
    echo $name

    if [ "$(ls -A $DIR)" ]; then  # skip if there were no parquets processed
     hadd $(echo ${name}_${channel})_merged.root *
     mv $(echo ${name}_${channel})_merged.root ../../../merged/2017/
    fi

    cd ../../..
  done
done
