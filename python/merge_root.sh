#!/bin/bash

set -e

mkdir -p roots

# process the cms data
for ch in rootfiles/* ; do
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    echo $name    
    hadd $name.root *
    mv $name.root ../../../roots/
    cd ../../..
  done
done
