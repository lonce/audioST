"""

"""
import tensorflow as tf
import numpy as np
import utils.spectreader as spectreader
import os
import time
import math

import utils.pickledModel as pickledModel

# get args from command line
import argparse
FLAGS = None
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--indir', type=str, help='directory holding TFRecords of data',  default='.') 
parser.add_argument('--outdir', type=str, help='output directory for logging',  default='.') 
parser.add_argument('--numClasses', type=int, help='number of classes in data', choices=[2,50], default=2) #default for testing
parser.add_argument('--checkpointing', type=int, help='0/1 - used for both saving and starting from checkpoints', choices=[0,1], default=0)
parser.add_argument('--checkpointPeriod', type=int, help='checkpoint every n batches', default=8) 

parser.add_argument('--freqbins', type=int, help='number of frequency bins in the spectrogram input', default=513) 
parser.add_argument('--numFrames', type=int, help='number of frames in the spectrogram input (must be divisible by 4)', default=424) 

parser.add_argument('--learning_rate', type=float, help='learning rate', default=.001) 
parser.add_argument('--batchsize', type=int, help='number of data records per training batch', default=8) #default for testing
parser.add_argument('--n_epochs', type=int, help='number of epochs to use for training', default=2) #default for testing
parser.add_argument('--keepProb', type=float, help='keep probablity for dropout before 1st fully connected layer during training', default=1.0) #default for testing
parser.add_argument('--batchnorm', type=int, help='0/1 - to batchnorm or not to batchnorm', choices=[0,1], default=1)


parser.add_argument('--freqorientation', type=str, help='freq as height or as channels', choices=["height","channels"], default="channels") #default for testing

parser.add_argument('--numconvlayers', type=int, help='number of convolutional layers', choices=[1,2],  default=2) #default for testing

parser.add_argument('--l1channels', type=int, help='Number of channels in the first convolutional layer', default=32) #default for testing
parser.add_argument('--l2channels', type=int, help='Number of channels in the second convolutional layer (ignored if numconvlayers is 1)', default=64) #default for testing
parser.add_argument('--fcsize', type=int, help='Dimension of the final fully-connected layer', default=32) #default for testing

parser.add_argument('--convRows', type=int, help='size of conv kernernel in freq dimension if orientation is height (otherwise ignored)',   default=5) #default for testing
parser.add_argument('--convColumns', type=int, help='size of conv kernernel in temporal dimension ',   default=5) #default for testing

parser.add_argument('--optimizer', type=str, help='optimizer', choices=["adam","gd"], default="gd") #default for testing
parser.add_argument('--adamepsilon', type=float, help='epsilon param for adam optimizer', default=.1) 

parser.add_argument('--learnCondition', type=str, help='when to learn', choices=["always","whenWrong"], default="always") #default for testing

parser.add_argument('--mtlnumclasses', type=int, help='if nonzero, train using secondary classes (which must be stored in TFRecord files', default=0)


FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed :  {0}'.format(FLAGS))



#HARD-CODED data-dependant parameters ------------------
#dimensions of image (pixels)
k_freqbins=FLAGS.freqbins

k_height=1						# default for freqs as channels
k_inputChannels=k_freqbins		# default for freqs as channels

if FLAGS.freqorientation == "height" :
	k_height=k_freqbins
	k_inputChannels=1

k_numFrames=FLAGS.numFrames

#number of samples for training and validation
k_numClasses=FLAGS.numClasses  #determines wether to read mini data set in data2 or full dataset in data50
validationSamples=8*k_numClasses
trainingSamples=32*k_numClasses


k_mtlnumclasses=FLAGS.mtlnumclasses #only matters if K_MTK is not 0

# ------------------------------------------------------
# Define paramaters for the training
learning_rate = FLAGS.learning_rate
k_batchsize = FLAGS.batchsize 
n_epochs = FLAGS.n_epochs #6  #NOTE: we can load from checkpoint, but new run will last for n_epochs anyway

# ------------------------------------------------------
# Define paramaters for the model 
K_NUMCONVLAYERS = FLAGS.numconvlayers

L1_CHANNELS=FLAGS.l1channels
L2_CHANNELS=FLAGS.l2channels
FC_SIZE = FLAGS.fcsize

k_downsampledHeight = 1			# default for freqs as channels
if FLAGS.freqorientation == "height" :
	# see https://www.tensorflow.org/api_guides/python/nn#convolution for calculating size from strides and padding
	k_downsampledHeight = int(math.ceil(math.ceil(k_height/2.)/2.))#   k_height/4  #in case were using freqs as y dim, and conv layers = 2
	print(':::::: k_downsampledHeight is ' + str(k_downsampledHeight))

k_downsampledWidth = k_numFrames/4 # no matter what the orientation - freqs as channels or as y dim
k_convLayerOutputChannels = L2_CHANNELS
if (K_NUMCONVLAYERS == 1) :
	k_downsampledWidth = k_numFrames/2
	k_convLayerOutputChannels = L1_CHANNELS
	if FLAGS.freqorientation == "height" :
		k_downsampledHeight = int(math.ceil(k_height/2.)) # k_height/2 #in case were using freqs as y dim, and conv layers = 1

print(':::::: k_downsampledHeight is ' + str(k_downsampledHeight))
print(':::::: k_downsampledWidth is ' + str(k_downsampledWidth))

K_ConvRows=1      # default for freqs as channels
if FLAGS.freqorientation == "height" :
	K_ConvRows=FLAGS.convRows
	
K_ConvCols=FLAGS.convColumns
k_ConvStrideRows=1
k_ConvStrideCols=1

k_poolRows = 1    # default for freqs as channels
k_poolStrideRows = 1  # default for freqs as channels
if FLAGS.freqorientation == "height" :
	k_poolRows = 2
	k_poolStrideRows = 2 



k_keepProb=FLAGS.keepProb

k_OPTIMIZER=FLAGS.optimizer
k_adamepsilon = FLAGS.adamepsilon

LEARNCONDITION = FLAGS.learnCondition

# ------------------------------------------------------
# Derived parameters for convenience (do not change these)
k_vbatchsize = min(validationSamples, k_batchsize)
k_numVBatches = validationSamples/k_vbatchsize
print(' ------- For validation, will run ' + str(k_numVBatches) + ' batches of ' + str(k_vbatchsize) + ' datasamples')

#ESC-50 dataset has 50 classes of 40 sounds each
k_batches_per_epoch = k_numClasses*40/k_batchsize
k_batchesPerLossReport= k_batches_per_epoch  #writes loss to the console every n batches
print(' ----------will write out report every ' + str(k_batchesPerLossReport) + ' batches')
#k_batchesPerLossReport=1 #k_batches_per_epoch

# Create list of paramters for serializing so that network can be properly reconstructed, and for documentation purposes
parameters={
	'k_height' : k_height, 
	'k_numFrames' : k_numFrames, 
	'k_inputChannels' : k_inputChannels, 
	'K_NUMCONVLAYERS' : K_NUMCONVLAYERS, 
	'L1_CHANNELS' : L1_CHANNELS, 
	'L2_CHANNELS' : L2_CHANNELS, 
	'FC_SIZE' : FC_SIZE, 
	'K_ConvRows' : K_ConvRows, 
	'K_ConvCols' : K_ConvCols, 
	'k_ConvStrideRows' : k_ConvStrideRows, 
	'k_ConvStrideCols' : k_ConvStrideCols, 
	'k_poolRows' : k_poolRows, 
	'k_poolStrideRows' : k_poolStrideRows, 
	'k_downsampledHeight' : k_downsampledHeight, 
	'k_downsampledWidth' : k_downsampledWidth,
	'freqorientation' : FLAGS.freqorientation
}
# ------------------------------------------------------
#Other non-data, non-model params
CHECKPOINTING=FLAGS.checkpointing
k_checkpointPeriod = FLAGS.checkpointPeriod  # in units of batches

INDIR = FLAGS.indir
OUTDIR = FLAGS.outdir

CHKPOINTDIR = OUTDIR + '/checkpoints' # create folder manually
CHKPTBASE =  CHKPOINTDIR + '/model.ckpt'	# base name used for checkpoints
LOGDIR = OUTDIR + '/log_graph'			#create folder manually
#OUTPUTDIR = i_outdir

NUM_THREADS = 4  #used for enqueueing TFRecord data 
#=============================================

def getImage(fnames, nepochs=None, mtlclasses=0) :
    """ Reads data from the prepaired *list* files in fnames of TFRecords, does some preprocessing 
    params:
    fnames - list of filenames to read data from
    nepochs - An integer (optional). Just fed to tf.string_input_producer().  Reads through all data num_epochs times before generating an OutOfRange error. None means read forever.
    """
    if mtlclasses : 
    	label, image, mtlabel = spectreader.getImage(fnames, nepochs, mtlclasses)
    else : 
    	label, image = spectreader.getImage(fnames, nepochs)

    #same as np.flatten
    # I can't seem to make shuffle batch work on images in their native shapes.
    image=tf.reshape(image,[k_freqbins*k_numFrames])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, k_numClasses))

    if mtlclasses :
    	mtlabel=tf.stack(tf.one_hot(mtlabel-1, mtlclasses))
    	return label, image, mtlabel
    else :
    	return label, image

def get_datafiles(a_dir, startswith):
    """ Returns a list of files in a_dir that start with the string startswith.
    e.g. e.g. get_datafiles('data', 'train-') 
    """ 
    return  [a_dir + '/' + name for name in os.listdir(a_dir)
            if name.startswith(startswith)]

def batch_norm(x, is_trainingP, scope):
	with tf.variable_scope(scope):
		return tf.layers.batch_normalization(x,
				axis=3, # is this right? - our conv2D returns NHWC ordering? 
				center=True, 
				scale=True, 
				training=is_trainingP,
				name=scope+"_bn")


#=============================================
# Step 1: Read in data

# getImage reads data for enqueueing shufflebatch, shufflebatch manages it's own dequeing 
# ---- First set up the graph for the TRAINING DATA
if k_mtlnumclasses : 
	target, data, mtltargets = getImage(get_datafiles(INDIR, 'train-'), nepochs=n_epochs, mtlclasses=k_mtlnumclasses)
	imageBatch, labelBatch, mtltargetBatch = tf.train.shuffle_batch(
	    [data, target, mtltargets], batch_size=k_batchsize,
	    num_threads=NUM_THREADS,
	    allow_smaller_final_batch=True, #want to finish an eposh even if datasize doesn't divide by batchsize
	    enqueue_many=False, #IMPORTANT to get right, default=False - 
	    capacity=1000,  #1000,
	    min_after_dequeue=500) #500
else :
	target, data  = getImage(get_datafiles(INDIR, 'train-'), n_epochs)
	imageBatch, labelBatch = tf.train.shuffle_batch(
	    [data, target], batch_size=k_batchsize,
	    num_threads=NUM_THREADS,
	    allow_smaller_final_batch=True, #want to finish an eposh even if datasize doesn't divide by batchsize
	    enqueue_many=False, #IMPORTANT to get right, default=False - 
	    capacity=1000,  #1000,
	    min_after_dequeue=500) #500


# ---- same for the VALIDATION DATA
# no need for mtl labels for validation
vtarget, vdata = getImage(get_datafiles(INDIR, 'validation-')) # one "epoch" for validation

#vimageBatch, vlabelBatch = tf.train.shuffle_batch(
#    [vdata, vtarget], batch_size=k_vbatchsize,
#    num_threads=NUM_THREADS,
#    allow_smaller_final_batch=True, #want to finish an eposh even if datasize doesn't divide by batchsize
#    enqueue_many=False, #IMPORTANT to get right, default=False - 
#    capacity=1000,  #1000,
#    min_after_dequeue=500) #500

vimageBatch, vlabelBatch = tf.train.batch(
    [vdata, vtarget], batch_size=k_vbatchsize,
    num_threads=NUM_THREADS,
    allow_smaller_final_batch=False, #want to finish an eposh even if datasize doesn't divide by batchsize
    enqueue_many=False, #IMPORTANT to get right, default=False - 
    capacity=1000)

# Step 2: create placeholders for features (X) and labels (Y)
# each lable is one hot vector.
# 'None' here allows us to fill the placeholders with different size batches (which we do with training and validation batches)
#X = tf.placeholder(tf.float32, [None,k_freqbins*k_numFrames], name= "X")
X = tf.placeholder(tf.float32, [None,k_freqbins*k_numFrames], name= "X")

if FLAGS.freqorientation == "height" :
	x_image = tf.reshape(X, [-1,k_height,k_numFrames,k_inputChannels]) 
else :
	print('set up reshaping for freqbins as channels')
	foo1 = tf.reshape(X, [-1,k_freqbins,k_numFrames,1]) #unflatten (could skip this step if it wasn't flattenned in the first place!)
	x_image = tf.transpose(foo1, perm=[0,3,2,1]) #moves freqbins from height to channel dimension

Y = tf.placeholder(tf.float32, [None,k_numClasses], name= "Y")  #labeled classes, one-hot
MTLY = tf.placeholder(tf.float32, [None,k_mtlnumclasses], name= "MTLY")  #labeled classes, one-hot 

# Step 3: create weights and bias
trainable=[]

#Layer 1
# 1 input channel, L1_CHANNELS output channels
isTraining=tf.placeholder(tf.bool, (), name= "isTraining") #passed in feeddict to sess.runs

w1=tf.Variable(tf.truncated_normal([K_ConvRows, K_ConvCols, k_inputChannels, L1_CHANNELS], stddev=0.1), name="w1")
trainable.extend([w1])

if (FLAGS.batchnorm==1) : 
	#convolve Wx (w/o adding bias) then relu 
	l1preactivation=tf.nn.conv2d(x_image, w1, strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') 
	bn1=batch_norm(l1preactivation, isTraining, "batch_norm_1")
	h1=tf.nn.relu(bn1, name="h1")
	# 2x2 max pooling
else : 
	# convolve and add bias    Wx+b
	b1=tf.Variable(tf.constant(0.1, shape=[L1_CHANNELS]), name="b1")
	trainable.extend([b1])
	l1preactivation=tf.nn.conv2d(x_image, w1, strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + b1
	h1=tf.nn.relu(l1preactivation, name="h1")

h1pooled = tf.nn.max_pool(h1, ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStrideRows, 2, 1], padding='SAME')


if K_NUMCONVLAYERS == 2 :
	#Layer 2
	#L1_CHANNELS input channels, L2_CHANNELS output channels
	w2=tf.Variable(tf.truncated_normal([K_ConvRows, K_ConvCols, L1_CHANNELS, L2_CHANNELS], stddev=0.1), name="w2")
	trainable.extend([w2])

	if (FLAGS.batchnorm==1) : 
		#convolve (w/o adding bias) then norm 
		l2preactivation= tf.nn.conv2d(h1pooled, w2, strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') 
		bn2=batch_norm(l2preactivation, isTraining, "batch_norm_2")
		h2=tf.nn.relu(bn2, name="h2")
	else :
		b2=tf.Variable(tf.constant(0.1, shape=[L2_CHANNELS]), name="b2")
		trainable.extend([b2])
		l2preactivation= tf.nn.conv2d(h1pooled, w2, strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + b2
		h2=tf.nn.relu(l2preactivation, name="h2")


	with tf.name_scope ( "Conv_layers_out" ):
		h2pooled = tf.nn.max_pool(h2, ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStrideRows, 2, 1], padding='SAME', name='h2_pooled')
		print('k_downsampledWidth = ' + str(k_downsampledWidth) + ', k_downsampledHeight = ' + str(k_downsampledHeight) + ', L2_CHANNELS = ' + str(L2_CHANNELS))
		print('requesting a reshape of size ' + str(k_downsampledWidth * k_downsampledHeight*L2_CHANNELS))
		convlayers_output = tf.reshape(h2pooled, [-1, k_downsampledWidth * k_downsampledHeight*L2_CHANNELS]) # to prepare it for multiplication by W_fc1

	#h2pooled is number of pixels / 2 / 2  (halved in size at each layer due to pooling)
	# check our dimensions are a multiple of 4
	if (k_numFrames%4) : # or ((FLAGS.freqorientation == "height") and k_height%4 )):
		print ('Error: width and height must be a multiple of 4')
		sys.exit(1)
else :
	convlayers_output = tf.reshape(h1pooled, [-1, k_downsampledWidth * k_downsampledHeight*L1_CHANNELS])

#now do a fully connected layer: every output connected to every input pixel of each channel
W_fc1 = tf.Variable(tf.truncated_normal([k_downsampledWidth * k_downsampledHeight * k_convLayerOutputChannels, FC_SIZE], stddev=0.1), name="W_fc1")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[FC_SIZE]) , name="b_fc1")

keepProb=tf.placeholder(tf.float32, (), name= "keepProb")
fc1preactivation = tf.matmul(tf.nn.dropout(convlayers_output, keepProb), W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fc1preactivation, name="h_fc1")

#Read out layer
W_fc2 = tf.Variable(tf.truncated_normal([FC_SIZE, k_numClasses], stddev=0.1), name="W_fc2")
b_fc2 = tf.Variable(tf.constant(0.1, shape=[k_numClasses]), name="b_fc2")

trainable.extend([W_fc1, b_fc1, W_fc2, b_fc2])

if k_mtlnumclasses : 
	#MTL Read out layer - This is the only part of the net that is different for the secondary classes
	mtlW_fc2 = tf.Variable(tf.truncated_normal([FC_SIZE, k_mtlnumclasses], stddev=0.1), name="mtlW_fc2")
	mtlb_fc2 = tf.Variable(tf.constant(0.1, shape=[k_mtlnumclasses]), name="mtlb_fc2")

	trainable.extend([mtlW_fc2, mtlb_fc2])

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
#could do a dropout here on h
logits_ = tf.matmul(h_fc1, W_fc2)
logits = tf.add(logits_ , b_fc2, name="logits")


if k_mtlnumclasses : 
	mtllogits = tf.matmul(h_fc1, mtlW_fc2) + mtlb_fc2

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# returns a 1D tensor of length batchsize
if LEARNCONDITION=="whenWrong" :
	summaryloss_primary_raw = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

	smpreds = tf.nn.softmax(logits=logits, name="softmax_preds")
	# argmax returns a batchsize tensor  of type int64, batchsize tensor of booleans
	# equal returns a batchsize tensor of type boolean
	wrong_preds = tf.not_equal(tf.argmax(smpreds, 1), tf.argmax(Y, 1))
	# ones where labe != max of softmax, tensor of length batchsize
	wrongMask = tf.cast(wrong_preds, tf.float32) # need numpy.count_nonzero(boolarr) :(
	summaryloss_primary = tf.multiply(summaryloss_primary_raw, wrongMask, name="wrongloss")
else :
	summaryloss_primary = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

meanloss_primary = tf.reduce_mean(summaryloss_primary)


if k_mtlnumclasses : 
	summaryloss_mtl = tf.nn.softmax_cross_entropy_with_logits(logits=mtllogits, labels=MTLY) 
	meanloss_mtl = tf.reduce_mean(summaryloss_mtl)
	meanloss=meanloss_primary+meanloss_mtl
else : 
	meanloss=meanloss_primary



#if k_mtlnumclasses :
#	meanloss = tf.assign(meanloss, meanloss_primary + meanloss_mtl) #training thus depends on MTLYY in the feeddict if k_mtlnumclasses  != 0
#else :
#	meanloss = tf.assign(meanloss, meanloss_primary)


# Step 6: define training op
# NOTE: Must save global step here if you are doing checkpointing and expect to start from step where you left off.
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer=None
if (k_OPTIMIZER == "adam") :
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=k_adamepsilon ).minimize(meanloss, var_list=trainable, global_step=global_step)
if (k_OPTIMIZER == "gd") :
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(meanloss, var_list=trainable, global_step=global_step)
assert(optimizer)

#Get the beta and gamma ops used for batchn ormalization since we have to update them explicitly during training
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print('extra update ops are ' + str(extra_update_ops))

#---------------------------------------------------------------
# VALIDATE
#--------------------------------------------------------------
# The nodes are used for running the validation data and getting accuracy scores from the logits
with tf.name_scope("VALIDATION"):
	softmax_preds = tf.nn.softmax(logits=logits, name="softmax_preds")
	# argmax returns a batchsize tensor  of type int64, batchsize tensor of booleans
	# equal returns a batchsize tensor of type boolean
	correct_preds = tf.equal(tf.argmax(softmax_preds, 1), tf.argmax(Y, 1))
	batchNumCorrect = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(

	# All this, just to feed a friggin float computed over several batches into a tensor we want to use for a summary
	validationtensor = tf.Variable(0.0, trainable=False, name="validationtensor")
	wtf = tf.placeholder(tf.float32, ())
	summary_validation = tf.assign(validationtensor, wtf)

#-----------------------------------------------------------------------------------
# These will be available to other programs that want to use this trained net.
tf.GraphKeys.USEFUL = 'useful'
tf.add_to_collection(tf.GraphKeys.USEFUL, X)    #input place holder
tf.add_to_collection(tf.GraphKeys.USEFUL, keepProb) #place holder
tf.add_to_collection(tf.GraphKeys.USEFUL, softmax_preds)
tf.add_to_collection(tf.GraphKeys.USEFUL, w1)
if (FLAGS.batchnorm==0) :
	tf.add_to_collection(tf.GraphKeys.USEFUL, b1)
tf.add_to_collection(tf.GraphKeys.USEFUL, w2)

if (FLAGS.batchnorm==0) :
	tf.add_to_collection(tf.GraphKeys.USEFUL, b2)
tf.add_to_collection(tf.GraphKeys.USEFUL, W_fc1)
tf.add_to_collection(tf.GraphKeys.USEFUL, b_fc1)
tf.add_to_collection(tf.GraphKeys.USEFUL, W_fc2)
tf.add_to_collection(tf.GraphKeys.USEFUL, b_fc2)



#-----------------------------------------------------------------------------------


# Run the validation set through the model and compute statistics to report as summaries
def validate(sess, printout=False) : 
	with tf.name_scope ( "summaries" ):
		# test the model
		total_correct_preds = 0

		try:
			for i in range(k_numVBatches):
				
				X_batch, Y_batch = sess.run([vimageBatch, vlabelBatch])
				batch_correct, predictions = sess.run([batchNumCorrect, softmax_preds], feed_dict ={ X : X_batch , Y : Y_batch, keepProb : 1., isTraining : False}) 
				
				total_correct_preds +=  batch_correct
				#print (' >>>>  Batch " + str(i) + ' with batch_correct = ' + str(batch_correct) + ', and total_correct is ' + str(total_correct_preds))

				if printout:
					print(' labels for batch:')
					print(Y_batch)
					print(' predictions for batch')
					print(predictions)
					# print num correct for each batch
					print(u'(Validation batch) num correct for batchsize of {0} is {1}'.format(k_vbatchsize , batch_correct))


			print (u'(Validation EPOCH) num correct for EPOCH size of {0} ({1} batches) is {2}'.format(validationSamples , i+1 , total_correct_preds))
			print('so the percent correction for validation set = ' + str(total_correct_preds/validationSamples))

			msummary = sess.run(mergedvalidation, feed_dict ={ X : X_batch , Y : Y_batch, wtf : total_correct_preds/validationSamples, keepProb : 1., isTraining : False}) #using last batch to computer loss for summary
			

		except Exception, e:
			print e

		return msummary


#--------------------------------------------------------------
#   Visualize with Tensorboard
# -------------------------------------------------------------

def create_train_summaries ():
		with tf.name_scope ( "train_summaries" ):
			tf.summary.scalar ( "mean_loss" , meanloss_primary)
			tf.summary.histogram ("w_1", w1)
			tf.summary.histogram ("l1preactivation", l1preactivation)
			tf.summary.histogram ("h_1", h1)
			tf.summary.histogram ("w_2", w2)
			tf.summary.histogram ("l2preactivation", l2preactivation)
			tf.summary.histogram ("h_2", h2)
			tf.summary.histogram ("w_fc1", W_fc1)
			tf.summary.histogram ("fc1preactivation", fc1preactivation)
			tf.summary.histogram ("h_fc1", h_fc1)
			tf.summary.histogram ("w_fc2", W_fc2)

			return tf.summary.merge_all ()

mergedtrain = create_train_summaries()

def create_validation_summaries ():
		with tf.name_scope ( "validation_summaries" ):
			#tf.summary.scalar ( "validation_correct" , batchNumCorrect)
			tf.summary.scalar ( "summary_validation", summary_validation)
			return tf.summary.merge_all ()

mergedvalidation = create_validation_summaries()

# --------------------------------------------------------------
# TRAIN
#---------------------------------------------------------------
def trainModel():

	with tf.Session() as sess:
		writer = tf.summary.FileWriter(LOGDIR)  # for logging
		saver = tf.train.Saver() # for checkpointing

		#### Must run local initializer if nepochs arg to getImage is other than None!
		#sess.run(tf.local_variables_initializer())
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

		#not doing it here, but global_step could have been initialized by a checkpoint
		if CHECKPOINTING :
			ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHKPTBASE))
		else :
			ckpt = False
		if ckpt and ckpt.model_checkpoint_path:
			print('Checkpointing restoring from path ' + ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			#only save graph if we are not starting run from a checkpoint
			writer.add_graph(sess.graph)

 
		initial_step = global_step.eval()
		print('initial step will be ' + str(initial_step)) # non-zero if check pointing
		batchcount=initial_step
		start_time = time.time()
		
		# Create a coordinator, launch the queue runner threads.
		coord = tf.train.Coordinator()
		enqueue_threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		
		try:
			batchcountloss = 0 #for reporting purposes
			while True: # for each batch, until data runs out
				if coord.should_stop():
					break

				
				
				if k_mtlnumclasses :
					X_batch, Y_batch, MTLY_batch = sess.run([imageBatch, labelBatch, mtltargetBatch])
					_, loss_batch, _nada = sess.run([optimizer, meanloss, extra_update_ops], feed_dict ={ X : X_batch , Y : Y_batch, keepProb : k_keepProb, MTLY : MTLY_batch, isTraining : True})   #DO WE NEED meanloss HERE? Doesn't optimer depend on it? 
				else :
					X_batch, Y_batch = sess.run([imageBatch, labelBatch])
					_, loss_batch, _nada = sess.run([optimizer, meanloss, extra_update_ops], feed_dict ={ X : X_batch , Y : Y_batch, keepProb : k_keepProb, isTraining : True})   #DO WE NEED meanloss HERE? Doesn't optimer depend on it?

				batchcountloss += loss_batch


				batchcount += 1
				if (not batchcount%k_batchesPerLossReport) :
					print('batchcount = ' + str(batchcount))
					avgBatchLoss=batchcountloss/k_batchesPerLossReport
					print(u'Average loss per batch {0}: {1}'.format(batchcount, avgBatchLoss))
					batchcountloss=0

					tsummary = sess.run(mergedtrain, feed_dict ={ X : X_batch , Y : Y_batch, keepProb : 1.0, isTraining : False }) #?? keep prob ??
					writer.add_summary(tsummary, global_step=batchcount)

					vsummary=validate(sess)
					writer.add_summary(vsummary, global_step=batchcount)


				if not (batchcount  % k_checkpointPeriod) :
					saver.save(sess, CHKPTBASE, global_step=batchcount)

		except tf.errors.OutOfRangeError, e:  #done with training epochs. Validate once more before closing threads
			# So how, finally?
			print('ok, let\'s validate now that we\'ve run ' + str(batchcount) + 'batches  ------------------------------')

			vsummary=validate(sess, False)
			writer.add_summary(vsummary, global_step=batchcount+1)


			coord.request_stop(e)

		except Exception, e:	
			print('train: WTF')
			print e

		finally :
			coord.request_stop()
			coord.join(enqueue_threads)
			writer.close()
		
		# grab the total training time
		totalruntime = time.time() - start_time
		print 'Total training time: {0} seconds'.format(totalruntime)
		print(' Finished!') # should be around 0.35 after 25 epochs

		print(' now save meta model')
		meta_graph_def = tf.train.export_meta_graph(filename=OUTDIR + '/my-model.meta')
		pickledModel.saveState(sess, trainable, parameters, OUTDIR + '/state.pickle') 

		print(' ===============================================================') 

#=============================================================================================
print(' ---- Actual parameters for this run ----')
print('INDIR : ' + INDIR)
print('k_freqbins : ' + str(k_freqbins) 
	+ '    ' + 'k_numFrames: ' + str(k_numFrames) )
#FLAGS.freqorientation, k_height, k_numFrames, k_inputChannels
print('FLAGS.freqorientation: ' + str(FLAGS.freqorientation) 
	+ ',   ' + 'k_height: ' + str(k_height) 
	+ ',   ' + 'k_numFrames: ' + str(k_numFrames) 
	+ ',   ' + 'k_inputChannels: ' + str(k_inputChannels))
#k_numClasses, validationSamples, trainingSamples
print('k_numClasses: ' + str(k_numClasses)
	+ ',   ' + 'validationSamples: ' + str(validationSamples)
	+ ',   ' + 'trainingSamples: ' + str(trainingSamples))
#learning_rate, k_keepProb, k_batchsize, n_epochs 
print('learning_rate: ' + str(learning_rate)
	+ ',   ' + 'k_keepProb: ' + str(k_keepProb)
	+ ',   ' + 'k_batchsize: ' + str(k_batchsize)
	+ ',   ' + 'n_epochs: ' + str(n_epochs))
#K_NUMCONVLAYERS,  L1_CHANNELS, L2_CHANNELS, FC_SIZE 
print('K_NUMCONVLAYERS: ' + str(K_NUMCONVLAYERS)
	+ ',   ' + 'L1_CHANNELS: ' + str(L1_CHANNELS)
	+ ',   ' + 'L2_CHANNELS: ' + str(L2_CHANNELS)
	+ ',   ' + 'FC_SIZE: ' + str(FC_SIZE))
#k_downsampledHeight, k_downsampledWidth , k_convLayerOutputChannels 
print('k_downsampledHeight: ' + str(k_downsampledHeight)
	+ ',   ' + 'k_downsampledWidth: ' + str(k_downsampledWidth)
	+ ',   ' + 'k_convLayerOutputChannels: ' + str(k_convLayerOutputChannels))
#K_ConvRows, K_ConvCols, k_ConvStrideRows, k_ConvStrideCols, k_poolRows, k_poolStrideRows 
print('K_ConvRows: ' + str(K_ConvRows)
	+ ',   ' + 'K_ConvCols: ' + str(K_ConvCols)
	+ ',   ' + 'k_ConvStrideRows: ' + str(k_ConvStrideRows)
	+ ',   ' + 'k_ConvStrideCols: ' + str(k_ConvStrideCols)
	+ ',   ' + 'k_poolRows: ' + str(k_poolRows)
	+ ',   ' + 'k_poolStrideRows : ' + str(k_poolStrideRows ))
if (k_OPTIMIZER == "adam") : 
	print('k_OPTIMIZER: ' + str(k_OPTIMIZER)
	+ ',   ' + 'k_adamepsilon: ' + str(k_adamepsilon))
else :
	print('k_OPTIMIZER: ' + str(k_OPTIMIZER))

print('LEARNCONDITION: ' + LEARNCONDITION)
print('batchnorm: ' + str(FLAGS.batchnorm))
print('k_mtlnumclasses: ' + str(k_mtlnumclasses))

#OUTDIR
print('OUTDIR: ' + str(OUTDIR))
print('CHECKPOINTING: ' + str(CHECKPOINTING))
print('     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv   ')
for x in trainable :
	print(x.name + ' : ' + str(x.get_shape()))
print('TOTAL number of parameters in the model is ' + str(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in trainable])))
print('     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv   ')

#=============================================================================================
# Do it
trainModel()
