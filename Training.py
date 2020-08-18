import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from data_pipeline import *
from Load_Data import *
from metrics import *



#Dynamic model design based on the vocabulary dimension 
def model(chunk,scope):
    #layer_size = int(0.8*vocab_size)
    chunk = tf.layers.Dense(500,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.initializers.he_normal(seed=1.5),bias_initializer=tf.initializers.constant(0),name=scope+'-l5')(chunk)
    chunk = tf.layers.Dense(500,activation=tf.nn.leaky_relu,use_bias=True,kernel_initializer=tf.initializers.he_normal(seed=None),bias_initializer=tf.initializers.constant(0),name=scope+'-l6')(chunk)
    chunk = tf.layers.Dense(500,activation=tf.nn.leaky_relu,use_bias=True,kernel_initializer=tf.initializers.he_normal(seed=None),bias_initializer=tf.initializers.constant(0),name=scope+'-l7')(chunk)
    chunk = tf.layers.Dense(500,activation=tf.nn.leaky_relu,use_bias=True,kernel_initializer=tf.initializers.he_normal(seed=None),bias_initializer=tf.initializers.constant(0),name=scope+'-l8')(chunk)
    
    chunk = tf.layers.Dense(500,activation=tf.nn.leaky_relu,use_bias=True,kernel_initializer=tf.initializers.he_normal(seed=None),bias_initializer=tf.initializers.constant(0),name=scope+'-l1')(chunk)
    out = tf.layers.Dense(11,kernel_initializer=tf.initializers.he_normal(seed=None),name=scope+'-befout')(chunk)
    out = tf.nn.softmax(out,name=scope+'-out')
    return out

