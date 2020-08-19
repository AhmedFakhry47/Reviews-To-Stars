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

	out       = Text_CNN(
				sequence_length =train_x.shape[1],
				vocab_size   = len(vocab_processor.vocabulary_),
				num_classes  = train_y.shape[1],
				window_width = 128,
				filter_sizes =[2,3,4],
				num_filters  =2,
				dropout      =False
				)

	loss      =  out.loss()
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss=loss,var_list=tf.trainable_variables())

	#Hyper parameters
	data_size = train_x.shape[0]
	batch_size= 48
	nepochs = 50
	nbatchs = math.floor(data_size/batch_size) + data_size%batch_size

	with tf.Session() as sess:
	    print('Started Training')
	    sess.run(tf.global_variables_initializer())
	    train_losses  =[]
	    devset_losses =[]
	    avg_recall    =[]
	    
	    for epoch in range(nepochs):
	        train = pipeline(batch_size,nbatchs,train_x,label_y)
	        epoch_loss = 0
	        k=0
	        for text,label in tqdm(train,total=nbatchs):
	            if(k==nbatchs): break
	            '''
	            if(text.any() == None):
	                break
	            '''
	            _,c = sess.run([optimizer,loss],feed_dict={reviews:text,stars:label})
	            epoch_loss += c
	            k+=1
	        train_losses.append(epoch_loss)
	        print('epoch_loss = ',epoch_loss/nbatchs)   
	        #devset_loss,avg_rec = evaluate_acc_loss_on_devset(test_x,test_y)
	        #devset_losses.append(devset_loss)
	        #avg_recall.append(avg_rec)
