#!/bin/bash

set -e

mkdir -p roots

# process the cms data
for ch in rootfiles/* ; do
  channel=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$ch")
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    echo $name
    hadd $name_$channel.root *
    mv $name_$channel.root ../../../roots/
    cd ../../..
  done
done
