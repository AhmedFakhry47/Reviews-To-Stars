from Load_Data import *
from metrics import *
from Model_utils import *
from data_pipeline import *

from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def evaluate_acc_loss_on_devset(test_x,test_y):

    with tf.Session() as test_sess:
        #Evaluate loss 
        devset_loss  = sess.run(out.loss,feed_dict={out.input_x:test_x,out.input_y:test_y,out.dropout_prob:1})
        #Evaluate Test Acc 
        avg_rec   	 = sess.run(out.accuracy,feed_dict={out.input_x:test_x,out.input_y:test_y,out.dropout_prob:1})
        #avg_rec = confusion(preds,test_y)
        return devset_loss,avg_rec*100

if __name__ == '__main__':
	data_x,data_y = load_data('data.csv')
	train_x,train_y,test_x,test_y,vocab_size = preprocess(data_x,data_y,train_test_ratio=0.9)
	print(train_y[:20])
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

	#Check points for step training_trial_step
	checkpoint_path   = "/home/enihcam/Downloads/github-repos/Reviews-To-Stars0/training_trial"
	checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
	if not os.path.exists(checkpoint_path):
		os.mkdir(checkpoint_path)
	
	train_saver = tf.train.Saver(max_to_keep=2)		
	with tf.Session() as sess:
	    print('Started Training')
	    sess.run(tf.global_variables_initializer())
	    train_losses  =[]
	    devset_losses =[]
	    avg_recall    =[]

	    for epoch in range(nepochs):
	        train = pipeline(batch_size,nbatchs,train_x,train_y)
	        #break
	        epoch_loss = 0
	        k=0
	        for text,label in tqdm(train,total=nbatchs):
	            if(k==nbatchs): break
	            '''
	            if(text.any() == None):
	                break
	            '''
	            _,c = sess.run([optimizer,out.loss],feed_dict={out.input_x:text,out.input_y:label,out.dropout_prob:0.5})
	            epoch_loss += c
	            k+=1
	        train_losses.append(epoch_loss)
	        print('epoch_loss = ',epoch_loss/nbatchs) 
	        print('\n')

	        avg_rec   	 = sess.run(out.accuracy,feed_dict={out.input_x:test_x,out.input_y:test_y,out.dropout_prob:1})
	        #devset_loss,avg_rec = evaluate_acc_loss_on_devset(test_x,test_y)
	        print('Test Acc :',avg_rec*100,' %')
	        #devset_losses.append(devset_loss)
	        avg_recall.append(avg_rec)
	        train_saver.save(sess,checkpoint_prefix)
