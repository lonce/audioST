#!/bin/bash                                                                                                                                                                      
# To store logs and see both stderr and stdout on the screen:                                                                                                                    
#    nohup ./run50.sh logs >>logs/multilog.txt 2>&1 &                                                                                                                                 
# Individual logs will also still get stored in their respective directories      
                                                                                              
source activate tflow2a
DATE=`date +%Y.%m.%d`
echo $DATE
#maindir=logs.$DATE
#mkdir $maindir

if [ $# -eq 0 ]
  then
    echo "please supply output directory as a command line argument"
    exit
fi

maindir=$1
mkdir $maindir

epsilon=1.0
optimizer=adam
learningrate=.01
orientationArray=(height)
layers=2
mtl=0

indir=data50Q

l1channels=0 # SET CONDITIONALLY BELOW
l2channelsArray=(64)
fcsize=32
bnArray=(0) 

for orientation in ${orientationArray[@]}
do
    if [ "$orientation" == "channels" ]
    then
	l1channels=2048
    else
	l1channels=32
    fi
    echo "l1 channels is  $l1channels"


    for l2channels in ${l2channelsArray[@]}
    do
        for bn in ${bnArray[@]}
        do
            #make output dir for paramter settings                                                                                                                               
            echo " -------       new batch run     --------"
            OUTDIR="$maindir/l1r_${l1channels}.l2_${l2channels}.fc_${fcsize}.or_${orientation}.bn_${bn}"
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #keep a copy of this run file                                                                                                                                        
            me=`basename "$0"`
            cp $me $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                                            
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"
            mkdir "$OUTDIR/stderr"
            # wrap python call in a string so we can do our fancy redirecting below                                                                                              
            runcmd='python DCNSoundClass.py --outdir $OUTDIR --checkpointing 1 --checkpointPeriod 500  --indir ${indir} '
            runcmd+=' --freqbins 513 --numFrames 424  --convRows 9 '
            runcmd+=' --numClasses 50 --batchsize 20 --n_epochs 100  --learning_rate ${learningrate} --batchnorm ${bn}'
            runcmd+=' --keepProb .5 --l1channels ${l1channels} --l2channels ${l2channels} --fcsize ${fcsize} --freqorientation ${orientation}  '
            runcmd+=' --numconvlayers ${layers} --adamepsilon ${epsilon} --optimizer ${optimizer} --mtlnumclasses ${mtl}'
                        # direct stdout and sterr from each run into their proper directories, but tww so we can still watch  
            echo "---------- now run!!!"
            eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR/stderr/stderr.log >&2)
        done
    done
done


