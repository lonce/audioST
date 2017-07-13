#!/bin/bash  
numshards=${1:-8}
echo "will run spect2TFRecords.2label.py with " $numshards " shards"
pyprog=`which spect2TFRecords.2label.py`
echo "will execute " $pyprog
python $pyprog --train_directory=./train --output_directory=./  --validation_directory=./validate --labels_file=mylabels.txt  --train_shards=$numshards --validation_shards=2 --num_threads=2