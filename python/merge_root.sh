#!/bin/bash

set -e


# process the cms data
for ch in rootfiles/* ; do
  for sample in $ch/* ; do
    cd $sample
    name=$(awk -F'/' '{ a = length($NF) ? $NF : $(NF-1); print a }' <<< "$sample")
    hadd $name.root *
    mv $name.root ..
  done
done

  # mv cms ../../data/
