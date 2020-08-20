import numpy as np
import PySimpleGUI as sg
import tensorflow  as tf
from Model_utils import *
from Load_Data import *
from tensorflow.keras.preprocessing import text, sequence

dummy_y = np.reshape(np.array([[1,0,0,0,0]]),(1,5	))
sg.theme('BluePurple')

layout = [[sg.Text('Number of stars:'), sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.Input(key='-IN-')],
          [sg.Button('Show'), sg.Button('Exit')]]

window = sg.Window('Reviews2Stars', layout)



#Check points for step training_trial_step
checkpoint_path   = "/home/enihcam/Downloads/github-repos/Reviews-To-Stars0/training_trial1"

'''
These parameters 
1870
37637
10
'''
max_length = 1870
vocab_size = 37637
num_classes= 5

def preprocess(data):
	tokenizer = text.Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(data)
	data_x          = tokenizer.texts_to_sequences(data)
	data_x          = sequence.pad_sequences(data_x, maxlen=max_length)
	data_x          = np.array(data_x)
	return data_x

data_x,_ = load_data('Reviews.csv')

out = Text_CNN(
	sequence_length = max_length,
	vocab_size      = vocab_size,
	num_classes     = num_classes,
	window_width    = 128,
	filter_sizes    = [2,3,4],
	num_filters     = 2,
	dropout         = False
)
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
train_saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
	train_saver.restore(sess,checkpoint_prefix)
	while True:  # Event Loop
	    event, values = window.read()
	    print(event, values)
	    if event == sg.WIN_CLOSED or event == 'Exit':
	        break
	    if event == 'Show':
	        # Update the "output" text element to be the value of "input" element
	        data = values['-IN-']
	        data = preprocess(data)
	        predictions = sess.run([out.predictions],feed_dict={out.input_x:data,out.input_y:dummy_y,out.dropout_prob:1})
	        window['-OUTPUT-'].update(str(predictions[0][0])+' of 5')

	window.close()