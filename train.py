from Load_Data import *
from metrics import *
from Model_utils import *
from data_pipeline import *

from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


if __name__ == '__main__':
	data_x,data_y = load_data('data.csv')
	train_x,train_y,test_x,test_y,vocab_size = preprocess(data_x,data_y,train_test_ratio=0.9)

	#Hyper parameters
	data_size = train_x.shape[0]
	batch_size= 48
	nepochs = 50
	nbatchs = math.floor(data_size/batch_size) + data_size%batch_size

	out = Text_CNN(
		sequence_length = train_x.shape[1],
		vocab_size      = vocab_size,
		num_classes     = train_y.shape[1],
		window_width    = 128,
		filter_sizes    = [2,3,4],
		num_filters     = 2,
		dropout         = False
	)

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=out.loss)

	with tf.Session() as sess:
	    print('Started Training')
	    sess.run(tf.global_variables_initializer())
	    train_losses  =[]
	    devset_losses =[]
	    avg_recall    =[]

	    for epoch in range(nepochs):
	        train = pipeline(batch_size,nbatchs,train_x,data_y)
	        epoch_loss = 0
	        k=0
	        for text,label in tqdm(train,total=nbatchs):
	            if(k==nbatchs): break
	            '''
	            if(text.any() == None):
	                break
	            '''
	            _,c,acc = sess.run([optimizer,out.loss,out.accuracy],feed_dict={out.input_x:text,out.input_y:label,out.dropout_prob:0.5})
	            epoch_loss += c
	            k+=1
	        train_losses.append(epoch_loss)
	        print('epoch_loss = ',epoch_loss/nbatchs) 
	        print('\n')
	        print('Training Acc : '+str(acc))
	        #devset_loss,avg_rec = evaluate_acc_loss_on_devset(test_x,test_y)
	        #devset_losses.append(devset_loss)
	        #avg_recall.append(avg_rec)
