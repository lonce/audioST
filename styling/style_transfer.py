
""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function
import sys 

import os
import time

import numpy as np
import tensorflow as tf

import pickledModel

# get args from command line
import argparse
FLAGS = []

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--content', type=str, help='name of file in content dir, w/o .ext'  ) 
parser.add_argument('--style', type=str, help='name of file in style dir, w/o .ext'  ) 
parser.add_argument('--noise', type=float, help='in range [0,1]', default=.5  ) 
parser.add_argument('--iter', type=int, help='number of iterations (on cpu, runtime is less than 1 sec/iter)', default=600  ) 
parser.add_argument('--alpha', type=float, help='amount to weight conent', default=10  ) 
parser.add_argument('--beta', type=float, help='amount to weight style', default=200  ) 
parser.add_argument('--randomize', type=int, help='0: use trained weights, 1: randomize model weights', choices=[0,1], default=0 ) 
parser.add_argument('--weightDecay', type=float, help='factor for L2 loss to keep vals in [0,255]',  default=.01 ) 

parser.add_argument('--outdir', type=str, help='for output images', default="." ) 
parser.add_argument('--stateFile', type=str, help='stored graph', default=None  ) 

FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed :  {0}'.format(FLAGS))

if any(v is None for v in vars(FLAGS).values()) :
    print('All args are required with their flags. For help: python style_transfer --help')
    sys.exit()


CHECKPOINTING=False

FILETYPE = ".tif"
# parameters to manage experiments
STYLE = FLAGS.style
CONTENT = FLAGS.content
STYLE_IMAGE = 'content/' + STYLE + FILETYPE
CONTENT_IMAGE = 'content/' + CONTENT + FILETYPE

  # This seems to be the paramter that really controls the balance between content and style
  # The more noise, the less content
NOISE_RATIO = FLAGS.noise # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['h1', 'h2']
W = [1.0, 2.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'h2'

#Relationship a/b is 1/20
ALPHA = FLAGS.alpha   #content
BETA = FLAGS.beta     #style

LOGDIR = FLAGS.outdir + '/log_graph'			#create folder manually
CHKPTDIR =  FLAGS.outdir + '/checkpoints'		# create folder manually
OUTPUTDIR = FLAGS.outdir

ITERS = FLAGS.iter
LR = 2.0

WEIGHT_DECAY=FLAGS.weightDecay

def _create_range_loss(im) : 
    over = tf.maximum(im-255, 0)
    under = tf.minimum(im, 0)
    out = tf.add(over, under)
    rangeloss = WEIGHT_DECAY*tf.nn.l2_loss(out)
    return rangeloss


def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    pdims=p.shape
    #print('p has dims : ' + str(pdims)) 
    coef = np.multiply.reduce(pdims)   # Hmmmm... maybe don't want to include the first dimension
    #this makes the loss 0!!!
    #return (1/4*coef)*tf.reduce_sum(tf.square(f-p))
    return tf.reduce_sum((f-p)**2)/(4*coef)


def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Hint: you'll first have to reshape F

        inputs: F: the tensor of all feature channels in a given layer
                N: number of features (channels) in the layer
                M: the total number of filters in each filter (length * height)

        F comes in as numchannels*length*height, and 
    """
        # We want to reshape F to be number of feaures (N) by the values in the feature array ( now represented in one long vector of length M) 

    Fshaped = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(Fshaped), Fshaped) # return G of size #channels x #channels


def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    horizdim = 1  # recall that first dimension of tensor is minibatch size
    vertdim = 2
    featuredim = 3



    # N - number of features
    N = a.shape[featuredim]  #a & g are the same shape
    # M - product of first two dimensions of feature map
    M = a.shape[horizdim]*a.shape[vertdim]

    #print(' N is ' + str(N)  + ', and M is ' + str(M))
    
    # This is 'E' from the paper and the homework handout.
    # It is a scalar for a single layer
    diff = _gram_matrix(a, N, M)-_gram_matrix(g, N, M)
    sq = tf.square(diff)
    s=tf.reduce_sum(sq)
    return (s/(4*N*N*M*M))
    

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    # E has one dimension with length equal to the number of layers
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]

    ###############################
    ## TO DO: return total style loss
    return np.dot(W, E)
    ###############################

def _create_losses(model, input_image, content_image, style_image):
    print('_create_losses')
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            # model[CONTENT_LAYER] is a relu op
            p = sess.run(model[CONTENT_LAYER])

        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        reg_loss = _create_range_loss(model['X'])

        ##########################################
        ## TO DO: create total loss. 
        ## Hint: don't forget the content loss and style loss weights
        total_loss = ALPHA*content_loss + BETA*style_loss + reg_loss
        ##########################################

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.name_scope ( "summaries" ):
        tf.summary.scalar ( "content loss" , model['content_loss'])
        tf.summary.scalar ( "style_loss" , model['style_loss'])
        tf.summary.scalar ( "total_loss" , model['total_loss'])
        # because you have several summaries, we should merge them all
        # into one op to make it easier to manage
        return tf.summary.merge_all()


def train(model, generated_image, initial_image):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.
    """
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run ( tf.global_variables_initializer ())
        print('initialize .....')
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        ###############################
        print('Do initial run to assign image')
        sess.run(generated_image.assign(initial_image))
        if CHECKPOINTING :
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHKPTDIR + '/checkpoint'))
        else :
            ckpt = False

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        step_time=start_time
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 100
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TO DO: obtain generated image and loss
                # following the optimazaiton step, calculate loss
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], 
                                                             model['summary_op']])
                
                ###############################
                #gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(sess.run(model['total_loss']))) #???????
                print('   Time: {}'.format(time.time() - step_time))
                step_time = time.time()

                filename = OUTPUTDIR + '/%d.tif' % (index)
                #pickledModel.save_image(np.transpose(gen_image[0][0]), filename)
                print('style_transfer: about to save image with shape ' + str(gen_image.shape))
                pickledModel.save_image(gen_image[0], filename)

                if (index + 1) % 20 == 0:
                    saver.save(sess, CHKPTDIR + '/style_transfer', index)

        print('   TOTAL Time: {}'.format(time.time() - start_time))
        writer.close()

#-----------------------------------

print('RUN MAIN')

model=pickledModel.load(FLAGS.stateFile, FLAGS.randomize)

print('MODEL LOADED')

model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

content_image = pickledModel.loadImage(CONTENT_IMAGE)
print('content_image shape is ' + str(content_image.shape))
print('content_image max is ' + str(np.amax(content_image) ))
print('content_image min is ' + str(np.amin(content_image) ))

#content_image = content_image - MEAN_PIXELS
style_image = pickledModel.loadImage(STYLE_IMAGE)
print('style_image max is ' + str(np.amax(style_image) ))
print('style_image min is ' + str(np.amin(style_image) ))
#style_image = style_image - MEAN_PIXELS

print(' NEXT, create losses')
model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                model["X"], content_image, style_image)
###############################
## TO DO: create optimizer
## model['optimizer'] = ...
model['optimizer'] =  tf.train.AdamOptimizer(LR).minimize(model['total_loss'], var_list=[model["X"]])
###############################
model['summary_op'] = _create_summary(model)

initial_image = pickledModel.generate_noise_image(content_image, NOISE_RATIO)
#def train(model, generated_image, initial_image):
train(model, model["X"], initial_image)

#if __name__ == '__main__':
#    main()
