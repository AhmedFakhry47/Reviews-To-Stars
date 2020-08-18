import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class Text_CNN():
	def __init__(self,sequence_length,window_width,window_height=[2,3,4],num_filters=6,num_classes,dropout=False):
		self.sequence_len  = sequence_length
		self.w_width 	   = window_width
		self.w_height 	   = window_height 
		self.n_filters 	   = num_filters
		self.n_classes     = num_classes

		self.input_x	   = tf.placeholder(tf.int32,[None,self.sequence_length],name='Word Vector')
		self.input_y	   = tf.placeholder(tf.float32,[None,num_classes],name='Output')
		self.dropout_prob  = tf.placeholder(tf.float32,name,name='dropout probability')

	#Dynamic model design based on the vocabulary dimension 
	def _model(self,):
		'''
		The architecture sequence is as follows:
		1-Embedding layer
		2-Convolution layer
		3-Max-Pooling Layers
		'''

		#First Layer 
		with tf.device('/cpu:0'),tf.name_scope('1-Embedding Layer'):
			Apply = tf.Variable(tf.random_uniform([self.w_width,self.w_height],-1.0,1.0),name='Gather-operation')
			self.embedded_chars = tf.nn.embedding_lookup(Apply,self.input_x)
			self.embedded_chars = tf.expand_dims(self.embedded_chars,-1)

		#Second Layer
		


