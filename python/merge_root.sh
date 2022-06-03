#!/bin/bash

set -e

mkdir -p roots

# process the cms data
for ch in rootfiles/* ; do
  for sample in $ch/* ; do
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    cd $sample
    hadd $name_$ch.root *
    mv $name_$ch.root ../../../roots/
    cd ../../..
  done
done
