#!/bin/bash  
#    nohup ./runstyle.sh >>styleout/2017.04.29/multilog.txt 2>&1 &     
# Individual logs will also still get stored in their respective directories 
source activate tflow2

statefile=logs.2017.05.14/mtl_16.or_height.epsilon_1.0/state.pickle
iter=200

noise=.2
rand=0
content=BeingRural5.0
style=agf5.0

python style_transfer.py --weightDecay 0 --content ${content} --style ${style}  --noise ${noise} --outdir testout  --stateFile ${statefile} --iter $iter --alpha 10 --beta 10 --randomize ${rand}
