"""
eg 
python testPickledModel.py  logs.2017.04.28/mtl_2.or_channels.epsilon_1.0/state.pickle  

"""
import tensorflow as tf
import numpy as np
import pickledModel

from PIL import TiffImagePlugin
from PIL import Image

# get args from command line
import argparse
FLAGS = None


VERBOSE=False
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('pickleFile', type=str, help='stored graph'  ) 
FLAGS, unparsed = parser.parse_known_args()

k_freqbins=257
k_width=856

styg = pickledModel.load(FLAGS.pickleFile)

print(' here we go ........')


def soundfileBatch(slist) :	
	return ([pickledModel.loadImage(name) for name in slist ])

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

im=np.empty([1,1,k_width,k_freqbins ])

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

with tf.Session() as sess:

	predictions=[]
	sess.run ( tf.global_variables_initializer ())

	#print('ok, all initialized')
	if 0 :
		print ('...GLOBAL_VARIABLES :')  #probalby have to restore from checkpoint first
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)

	if 0 :
		for v in ["s_w1:0", "s_b1:0", "s_w2:0", "s_b2:0", "s_W_fc1:0", "s_b_fc1:0", "s_W_fc2:0", "s_b_fc2:0"] :
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v)))


	if 1 :
		for v in ["s_h1:0"] :
			#im = np.reshape(np.transpose(rimages[6]), [1,k_width*k_freqbins ])
			im=rimages[6]
			print('assigning input variable an image with shape  ' + str(im.shape))
			sess.run(styg["X"].assign(im)) #transpose to make freqbins channels
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v)))



	print('predictions are : ')
	for im_ in rimages :
		#im = np.reshape(np.transpose(im_), [1,k_width*k_freqbins ])
		im=im_
		sess.run(styg["X"].assign(im)) #transpose to make freqbins channels
		prediction = sess.run(styg["softmax_preds"])
		print(str(prediction[0]))  
		#predictions.extend(prediction[0])


	#pickledModel.save_image(np.transpose(im, [0,3,2,1])[0,:,:,0],'fooimage.tif')
	pickledModel.save_image(im[0,:,:,:],'fooimage.tif')

