#!/bin/bash

set -e


# process the cms data
for ch in rootfiles/* ; do
  for sample in ch/* ; do
    echo $sample
    # #generate pytorch data files from pkl files
    # python3 dataset.py --data cms --dataset $sample \
    #   --processed_dir $sample/processed --num-files-merge 1 --num-proc 1
  done
done

  # mv cms ../../data/
