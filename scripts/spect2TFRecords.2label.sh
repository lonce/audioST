#!/bin/bash  
numshards=${1:-8}
python spect2TFRecords.2label.py --train_directory=./train --output_directory=./  --validation_directory=./validate --labels_file=mylabels.txt  --train_shards=$numshards --validation_shards=2 --num_threads=2