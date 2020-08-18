from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math
import os

def navigate(string):
	'''
	-A function navigates through the current working directory
	-Returns the first file that meets the conditions
	'''
	current_dir = os.getcwd()
	for dirname,_,files in os.walk(current_dir):
		for file in files:
			if file == dataname:
				return os.path.join(dirname,file)

def load_data(dataname):
	'''
	The whole point of this function is to load 
	'''
	data_dir 	= navigate(dataname)
	data 		= pd.read_csv('Data/data.csv',usecols=['Stars','Review'])
	vectorizer = CountVectorizer(lowercase=True,min_df=100)
	vectorizer.fit(dataset['Review'].to_list()) 
	data_X = vectorizer.transform(dataset['Review'].to_list())
	data_Y = np.array(dataset['Stars'])

	return data_X,data_Y
	
