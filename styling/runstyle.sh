#!/bin/bash                                                                                                                                                        
#    nohup ./runstyle.sh >>styleout/2017.05.02/multilog.txt 2>&1 &                                                                                                 
# Individual logs will also still get stored in their respective directories                                                                                       
source activate tflow2
DATE=`date +%Y.%m.%d`
maindir=styleout/$DATE
mkdir $maindir

statefile=testmodel/state.pickle
iter=3000
alpha=10
betaArray=(2 10)
noiseArray=(.2 .7)
rand=0
#contentArray=(BeingRural5.0 agf5.0 Superstylin5.0 roosters5.0 Nancarrow5.0 Toys5.0 inc5.0 sheepfarm5.0)
#styleArray=(BeingRural5.0 agf5.0 Superstylin5.0 roosters5.0 Nancarrow5.0 Toys5.0 inc5.0 sheepfarm5.0)
contentArray=(Superstylin5.0 agf5.0 wavenetbabble5.0 Toys5.0 inc5.0 Nancarrow5.0)
styleArray=(Superstylin5.0 agf5.0 wavenetbabble5.0 Toys5.0 inc5.0 Nancarrow5.0)

for noise in ${noiseArray[@]}
do
 for beta in ${betaArray[@]}
 do
      for content in ${contentArray[@]}
      do
	  for style in ${styleArray[@]}
	  do

	      if [ "$style" == "$content" ]
	      then
		  continue
	      fi
	      #make output dir for paramter settings                                                                                                        
	      echo " -------       new batch run     --------"
	      OUTDIR="$maindir/content_${content}.style_${style}.beta_${beta}.noise_${noise}"
	      mkdir $OUTDIR
	      echo "outdir is " $OUTDIR

	      #make subdirs for logging and checkpoints                                                                                                     

	      mkdir "$OUTDIR/log_graph"
	      mkdir "$OUTDIR/checkpoints"

	      runcmd='python style_transfer.py --content ${content} --style ${style}  --noise ${noise} --outdir $OUTDIR '
	      runcmd+='--stateFile ${statefile} --iter $iter --alpha ${alpha} --beta ${beta} --randomize ${rand}'
	      eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR.stderr.log >&2)
	  done
      done
 done
done
