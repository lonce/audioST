#
#
#Morgans great example code:
#https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
#
# GitHub utility for freezing graphs:
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#
#https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants


import tensorflow as tf
import numpy as np

#global variables 
g_st_saver=None
g_chkptdir=None
g_trainedgraph=None

VERBOSE=1


#-------------------------------------------------------------

def load(meta_model_file, restore_chkptDir) :

	global g_st_saver
	global g_chkptdir
	global g_trainedgraph

	g_st_saver = tf.train.import_meta_graph(meta_model_file)
	# Access the graph
	g_trainedgraph = tf.get_default_graph()

	with tf.Session() as sess:
		g_chkptdir=restore_chkptDir # save in global for use during initialize
		#g_st_saver.restore(sess, tf.train.latest_checkpoint(restore_chkptDir))



	return g_trainedgraph, g_st_saver

def initialize_variables(sess) :
	g_st_saver.restore(sess, tf.train.latest_checkpoint(g_chkptdir))

	tf.GraphKeys.USEFUL = 'useful'
	var_list = tf.get_collection(tf.GraphKeys.USEFUL)

	#print('var_list[3] is ' + str(var_list[3]))
	

	#JUST WANTED TO TEST THIS TO COMPARE TO STYLE MODEL CODE
	# Now get the values of the trained graph in to the new style graph
	#sess.run((g_trainedgraph.get_tensor_by_name("w1:0")).assign(var_list[3]))
	#sess.run(g_trainedgraph.get_tensor_by_name("b1:0").assign(var_list[4]))
	#sess.run(g_trainedgraph.get_tensor_by_name("w2:0").assign(var_list[5]))
	#sess.run(g_trainedgraph.get_tensor_by_name("b2:0").assign(var_list[6]))

	#sess.run(g_trainedgraph.get_tensor_by_name("W_fc1:0").assign(var_list[7]))
	#sess.run(g_trainedgraph.get_tensor_by_name("b_fc1:0").assign(var_list[8]))
	#sess.run(g_trainedgraph.get_tensor_by_name("W_fc2:0").assign(var_list[9]))
	#sess.run(g_trainedgraph.get_tensor_by_name("b_fc2:0").assign(var_list[10]))

	
