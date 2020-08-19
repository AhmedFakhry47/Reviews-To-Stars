import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class Text_CNN():
	def __init__(self,sequence_length,vocab_size,num_classes,filter_sizes=[2,3,4],num_filters=2,dropout=False):
		'''
		Sequence length : number of words per each example
		Vocab_size      : After mapping each word to an integer 
		window width    : is the embedding size

		'''
		self.num_words	   = sequence_length
		self.n_classes     = num_classes
		self.vocab_size    = vocab_size
		self.w_width 	   = window_width 
		self.filter_sizes  = filter_sizes
		self.n_filters 	   = num_filters
		

		self.input_x	   = tf.placeholder(tf.int32,[None,self.sequence_length],name='Word Vector')
		self.input_y	   = tf.placeholder(tf.float32,[None,num_classes],name='Output')
		self.dropout_prob  = tf.placeholder(tf.float32,name,name='dropout probability')

	#Dynamic model design based on the vocabulary dimension 
	def _model(self,):
		'''
		The architecture sequence is as follows:
		1-Embedding layer
		2-Convolution layer for each window_height
		3-Max-Pooling Layers
		'''

		#First Block: Embedding Block
		with tf.device('/cpu:0'),tf.name_scope('Embedding Layer'):
			embedded_weights	= tf.Variable(tf.random_uniform([self.vocab_size,self.w_width],-1.0,1.0),name='Gather-operation')
			self.embedded_chars = tf.nn.embedding_lookup(embedded_weights,self.input_x)
			self.embedded_chars = tf.expand_dims(self.embedded_chars,-1) #Output of embedding block

		#Second Block: Convolution Block
		for i,filter_size in enumerate(self.filter_sizes):
			with tf.name_scope('Convolution Layers # '+str(i)):

				#Weights initialization
				filter_shape = [filter_size,self.w_width,1,self.n_filters]
				weights = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='Conv-Weights _num# '+str(i))
				biases  = tf.Variable(tf.constant(0.1,shape=[num_filters]))

				#Convolution
				conv    = tf.nn.conv2d(self.embedded_chars,weights,strides=[1,1,1,1],padding='VALID',name='Conv #'+str(i))

				#Activation 
				act_out = tf.nn.relu(tf.nn.bias_add(conv,biases),name='Activation #'+str(i))

				#Max Pooling
				Pooling    = tf.nn.max_pool(act_out,ksize=[])







