#!/bin/bash  
numshards=${1:-8}
echo "will run spect2TFRecords with " $numshards " shards"
python spect2TFRecords.py --train_directory=./train --output_directory=./  --validation_directory=./validate --labels_file=mylabels.txt  --train_shards=$numshards --validation_shards=2 --num_threads=2
