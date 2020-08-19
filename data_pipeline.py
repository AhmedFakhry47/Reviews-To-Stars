import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing import text, sequence
import numpy as np

def preprocess(data_x,data_y,train_test_ratio=0.9):

    #Build vocabulary
    max_length = max([len(text.split(" ")) for text in data_x])

    vectorizer = CountVectorizer(lowercase=True,max_df=100)
    vectorizer.fit(data_x)
    vocab_size = len(vectorizer.vocabulary_)

    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data_x)
    data_x          = tokenizer.texts_to_sequences(data_x)
    data_x          = sequence.pad_sequences(data_x, maxlen=max_length)
    data_x          = np.array(data_x)
    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
    #data_x          = np.array(list(vocab_processor.fit_transform(data_x)))
    data_y          = np.array(data_y)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(data_y.shape[0]))
    x_shuffled      = data_x[shuffle_indices]
    y_shuffled      = data_y[shuffle_indices]


    #Divide data
    data_length     = len(x_shuffled)
    dividing_index  = int(train_test_ratio*data_length)

    train_x         = x_shuffled[:dividing_index]
    train_y         = y_shuffled[:dividing_index]
    test_x          = x_shuffled[dividing_index:]
    test_y          = y_shuffled[dividing_index:]

    return train_x,train_y,test_x,test_y,vocab_size 


def hotkey(arr):
    temp = np.zeros((arr.size,arr.max()+1))
    temp[np.arange(arr.size),arr] = 1    
    return temp
    
def pipeline(batch_size,nbatchs,data,label):
    i = 0
    while (True):
        if(i >= nbatchs):
            print('Done')
            yield None,None
            break
        c_data  = data[i*batch_size:((i+1)*batch_size)]
        c_label = label[i*batch_size:((i+1)*batch_size)]

        #c_data  = data[i*batch_size:((i+1)*batch_size)].toarray()
        #c_label = hotkey(label[i*batch_size:((i+1)*batch_size)])
        yield c_data,c_label
        i+1