from Load_Data import *
from metrics import *
from Model_utils import *
from data_pipeline import *


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


if __name__ == '__main__':
	data_x,data_y = load_data('data.csv')
	train_x,train_y,test_x,test_y,vocab_processor = preprocess(data_x,data_y,train_test_ratio=0.9)

downhill  = model(reviews,"My-Model")

#loss      = tf.losses.softmax_cross_entropy(onehot_labels=stars,logits=downhill,scope="Loss-Function")
loss      = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(stars,1), logits=downhill,name='Loss'))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss=loss,var_list=tf.trainable_variables())