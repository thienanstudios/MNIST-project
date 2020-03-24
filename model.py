# defining neural network architecture
# building the model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# y = mx + b
# output = weight * input + bias

model = tf.keras.Sequential(layers = None, name = None)

def sigmoid(z) : 
  return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(x) * (1 - sigmoid(x))




# layers and values
input_layer = 784
hidden1 = 512
hidden2 = 256
hidden3 = 128
output = 10


# tensors for images
tx = tf.placeholder("float", shape=(None, input_layer))
ty = tf.placeholder("float", shape=(None, output))
# some images will be dropped out, random sample size
# random images get dropped from the subset
keep_prob = tf.placeholder(tf.float32)

#constants
iterations = 1000
batchsize = 100     #subset, choose size
learnrate = 1e-4    # learning rate
dropout = 0.2



# do not need weights for input
# n - 1 number of weights
weights = {
  'weight1' : tf.Variable(tf.truncated_normal([input_layer, hidden1], stddev = 0.1)),
  'weight2' : tf.Variable(tf.truncated_normal([hidden1, hidden2], stddev = 0.1)),
  'weight3' : tf.Variable(tf.truncated_normal([hidden2, hidden3], stddev = 0.1)),
  'weightOut' : tf.Variable(tf.truncated_normal([hidden3, output], stddev = 0.1)),
}


# biases
biases = {
  'bias1' : tf.Variable(tf.constant(0.1, shape = [hidden1])),
  'bias2' : tf.Variable(tf.constant(0.1, shape = [hidden2])),
  'bias3' : tf.Variable(tf.constant(0.1, shape = [hidden3])),
  'biasOut' : tf.Variable(tf.constant(0.1, shape = [output])),
}


# create layers
layer1 = tf.add(tf.matmul(tx, weights['weight1']), biases['bias1'])
layer2 = tf.add(tf.matmul(layer1, weights['weight2']), biases['bias2'])
layer3 = tf.add(tf.matmul(layer2, weights['weight3']), biases['bias3'])
layerdrop = tf.nn.dropout(layer3, dropout)
outputLayer = tf.matmul(layer3, weights['weightOut']) + biases['biasOut']


# accurracy, # correct, loss function, TRAININGSTEP

lossfunction = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(
    labels = ty, logits = outputLayer
  )
)

trainingstep = tf.train.AdamOptimizer(0.0001).minimize(lossfunction)