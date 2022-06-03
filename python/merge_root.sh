#!/bin/bash

# run after running dataframe2root.py to merge the rootfiles

set -e

mkdir -p roots

# process the cms data
for ch in rootfiles/* ; do
  channel=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$ch")
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    echo $name
    hadd $(echo ${name}_${channel}).root *
    mv $(echo ${name}_${channel}).root ../../../roots/
    cd ../../..
  done
done
