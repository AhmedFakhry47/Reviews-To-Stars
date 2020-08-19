from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import math
import os
import re


def int_to_label(number,base):
	vector = np.zeros(base)
	vector[number-1] = 1
	return vector

def text_flourish(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def navigate(dataname):
	'''
	-A function navigates through the current working directory
	-Returns the first file that meets the conditions
	'''
	current_dir = os.getcwd()
	for dirname,_,files in os.walk(current_dir):
		for file in files:
			if file == dataname:
				return os.path.join(dirname,file)

def pad_sentences(sentences,lengths):
	'''
	For a better implementation later
	'''
	max_len = max(lengths)
	for i,sent in enumerate(sentences):
		current_length = len(sent)
		diff 		   = max_len-current_length
		sentences[i]   = sent + ['PAD']*diff
	return

def load_data(dataname):
	'''
	Load data and vectorize the labels
	'''
	data_dir 	= navigate(dataname)
	dataset 	= pd.read_csv(data_dir,usecols=['Stars','Review'])

	data_x = []
	data_y = []

	for i in dataset['Stars'].unique():
		current = dataset['Review'].loc[dataset['Stars']==i]
		label   = int_to_label(i,10)
		for current_x in current:
			text = text_flourish(current_x)
			data_x.append(text)
			data_y.append(label)

	#data_x = pad_sentences(data_x)
	return data_x,data_y
