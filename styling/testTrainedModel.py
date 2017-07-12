"""
eg 
python testModel.py  logs.2017.04.28/mtl_2.or_channels.epsilon_1.0/my-model.meta  logs.2017.04.28/mtl_2.or_channels.epsilon_1.0/checkpoints/

"""
import tensorflow as tf
import numpy as np
import trainedModel

from PIL import TiffImagePlugin
from PIL import Image

# get args from command line
import argparse
FLAGS = None

VERBOSE=False
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('metamodel', type=str, help='stored graph'  ) 
parser.add_argument('checkptDir', type=str, help='the checkpoint directory from where the latest checkpoint will be read to restore values for variables in the graph'  ) 
FLAGS, unparsed = parser.parse_known_args()

k_freqbins=257
k_width=856

g, savior = trainedModel.load(FLAGS.metamodel, FLAGS.checkptDir)


#vnamelist =[n.name for n in tf.global_variables()]
if VERBOSE : 
	vnamelist =[n.name for n in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
	print('TRAINABLE vars:')
	for n in vnamelist :
		print(n)


#opslist  = [n.name for n in g.get_operations()] 
#print('----Operatios in graph are : ' + str(opslist))
tf.GraphKeys.USEFUL = 'useful'

if VERBOSE : 
	print ('...and useful :')  #probalby have to restore from checkpoint first
	all_vars = tf.get_collection(tf.GraphKeys.USEFUL)
	for v in all_vars:
		print(v)

#
#print(' here we go ........')


var_list = tf.get_collection(tf.GraphKeys.USEFUL)

####tf.add_to_collection(tf.GraphKeys.USEFUL, X)    #input place holder
####tf.add_to_collection(tf.GraphKeys.USEFUL, keepProb) #place holder
####tf.add_to_collection(tf.GraphKeys.USEFUL, softmax_preds)
####tf.add_to_collection(tf.GraphKeys.USEFUL, h1)
####tf.add_to_collection(tf.GraphKeys.USEFUL, h2)

#X = g.get_tensor_by_name('X/Adam:0')# placeholder for input
#X = tf.placeholder(tf.float32, [None,k_freqbins*k_width], name= "X")
X=var_list[0]
#print('X is ' + str(X))

#keepProb = g.get_tensor_by_name('keepProb')
#keepProb=tf.placeholder(tf.float32, (), name= "keepProb")
keepProb=var_list[1]
#print('keepProb is ' + str(keepProb))


softmax_preds=var_list[2]
assert softmax_preds.graph is tf.get_default_graph()

def soundfileBatch(slist) :
	# The training network scales to 255 and then flattens before stuffing into batches
	return [np.array(Image.open(name).point(lambda i: i*255)).flatten() for name in slist ]


#just test the validation set 
#Flipping and scaling seem to have almost no effect on the clasification accuracy
rimages=soundfileBatch(['data2/validate/205 - Chirping birds/5-242490-A._11_.tif',
	'data2/validate/205 - Chirping birds/5-242491-A._12_.tif',
	'data2/validate/205 - Chirping birds/5-243448-A._14_.tif',
	'data2/validate/205 - Chirping birds/5-243449-A._15_.tif',
	'data2/validate/205 - Chirping birds/5-243450-A._15_.tif',
	'data2/validate/205 - Chirping birds/5-243459-A._13_.tif',
	'data2/validate/205 - Chirping birds/5-243459-B._13_.tif',
	'data2/validate/205 - Chirping birds/5-257839-A._10_.tif',
	'data2/validate/101 - Dog/5-203128-A._4_.tif',
	'data2/validate/101 - Dog/5-203128-B._5_.tif',
	'data2/validate/101 - Dog/5-208030-A._9_.tif',
	'data2/validate/101 - Dog/5-212454-A._4_.tif',
	'data2/validate/101 - Dog/5-213855-A._4_.tif',
	'data2/validate/101 - Dog/5-217158-A._2_.tif',
	'data2/validate/101 - Dog/5-231762-A._1_.tif',
	'data2/validate/101 - Dog/5-9032-A._12_.tif',
	])

#rimages=np.random.uniform(0.,1., (3,k_freqbins*k_width))


#print('got my image, ready to run!')

#Z = tf.placeholder(tf.float32, [k_freqbins*k_width], name= "Z")
#Y=tf.Variable(tf.truncated_normal([k_freqbins*k_width], stddev=0.1), name="Y")
#Y=tf.assign(Y,Z)

#with tf.Session() as sess:
#	sess.run ( tf.global_variables_initializer ())
#	foo = sess.run(Y, feed_dict={Z: rimage})
print(' here we go ........')

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

with tf.Session() as sess:
	#sess.run ( tf.global_variables_initializer ())
	#savior.restore(sess, tf.train.latest_checkpoint(FLAGS.checkptDir))
	trainedModel.initialize_variables(sess)
	if 0 :
		print ('...GLOBAL_VARIABLES :')  #probalby have to restore from checkpoint first
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)

	if 0 :
		for v in ["w1:0", "b1:0", "w2:0", "b2:0", "W_fc1:0", "b_fc1:0", "W_fc2:0", "b_fc2:0"] :
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v)))

	if 1 :
		for v in ["h1:0"] :
			im = np.reshape(rimages[6], [1,k_width*k_freqbins ])
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v), feed_dict ={ X : im,  keepProb : 1.0 }))


	print('predictions are : ')
	for im_ in rimages :
		im = np.reshape(im_, [1,k_width*k_freqbins ])
		prediction = sess.run(softmax_preds, feed_dict ={ X : im,  keepProb : 1.0 })
		print(str(prediction[0]))


	# Run the standard way .... in batches
	#predictions = sess.run(softmax_preds, feed_dict ={ X : rimages ,  keepProb : 1.0 })
	#print('predictions are : ')
	#print(str(predictions))

