import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class Text_CNN():
	def __init__(self,sequence_length,window_width,vocab_size,num_classes,filter_sizes=[2,3,4],num_filters=2,dropout=False):
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
		self.dropout       = dropout

		self.input_x	   = tf.placeholder(tf.int32,[None,self.num_words],name='Word_Vector')
		self.input_y	   = tf.placeholder(tf.float32,[None,self.n_classes],name='Output')
		self.dropout_prob  = tf.placeholder(tf.float32,name='dropout_probability')

		self.model()

	#Dynamic model design based on the vocabulary dimension 
	def model(self):
		'''
		The architecture sequence is as follows:
		1-Embedding layer
		2-Convolution layer for each window_height
		3-Max-Pooling Layers
		'''

		#First Block: Embedding Block
		with tf.device('/cpu:0'),tf.name_scope('Embedding_Layer'):
			embedded_weights	= tf.Variable(tf.random_uniform([self.vocab_size,self.w_width],-1.0,1.0),name='Gather-operation')
			self.embedded_chars = tf.nn.embedding_lookup(embedded_weights,self.input_x)
			self.embedded_chars = tf.expand_dims(self.embedded_chars,-1) #Output of embedding block

		pooled_features=[]
		#Second Block: Convolution Block
		for i,filter_size in enumerate(self.filter_sizes):
			with tf.name_scope('Convolution_Layers_'+str(i)):

				#Weights initialization
				filter_shape = [filter_size,self.w_width,1,self.n_filters]
				weights = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='Conv-Weights_num_'+str(i))
				biases  = tf.Variable(tf.constant(0.1,shape=[self.n_filters]))

				#Convolution
				conv    = tf.nn.conv2d(self.embedded_chars,weights,strides=[1,1,1,1],padding='VALID',name='Conv_'+str(i))

				#Activation 
				act_out = tf.nn.relu(tf.nn.bias_add(conv,biases),name='Activation_'+str(i))

				#Max Pooling
				Pooling = tf.nn.max_pool(act_out,ksize=[1,self.num_words-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='Max_Pooling_'+str(i))
				pooled_features.append(Pooling)

		tot_nfilters 	  = self.n_filters * len(self.filter_sizes)
		self.all_features = tf.concat(pooled_features,3)
		self.all_features = tf.reshape(self.all_features,[-1,tot_nfilters])

	
		#with tf.name_scope("dropout"):
		#	self.all_features =  tf.nn.dropout(self.all_features,self.dropout_prob)

		with tf.name_scope("Output"):
			out_weights = tf.Variable(tf.truncated_normal([tot_nfilters , self.n_classes], stddev=0.1), name='Weights_of_output')
			out_biases  = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name='Output_nodes')

		#	print(self.input_x)
			self.scores 	 = tf.nn.xw_plus_b(self.all_features, out_weights, out_biases, name='Output_nodes')
			self.predictions = tf.argmax(self.scores, 1, name='Output-Activated')

		with tf.name_scope("dropout"):
			self.all_features =  tf.nn.dropout(self.all_features,self.dropout_prob)
			
		#print(self.scores)
		with tf.name_scope("loss"):
			losses 	  = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses)

		with tf.name_scope("acc"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")	

		return self.predictions







