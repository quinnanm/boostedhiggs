#!/bin/bash

# run after running dataframe2root.py to merge the rootfiles

set -e

mkdir -p merged/2017

# process the cms data
for ch in rootfiles/* ; do
  channel=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$ch")
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    echo $name

    if [ "$(ls -A $DIR)" ]; then
     hadd $(echo ${name}_${channel})_merged.root *
     mv $(echo ${name}_${channel})_merged.root ../../../merged/2017/
    fi

    cd ../../..
  done
done
