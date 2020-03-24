import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model as model
import random as rand


# getting the data set from tensorflow
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()





print(x_train.shape)


# normalize dataset
# max value is 255 because 255 is the rgb value
x_train = x_train / 255.0
x_test = x_test / 255.0


# comparing predictions with the actual value
predictions = tf.cast(tf.equal(tf.argmax(model.outputLayer, 1), tf.argmax(y_train, 1)), tf.float32)
accuracy = tf.reduce_mean(predictions)


# creating a session to train on
initialize = tf.global_variables_initializer()
session = tf.Session()
session.run(initialize)


xTrainLength = len(x_train)

# actual training
for j in range(model.iterations):
  #randNum = rand.random()
  #randNum = int(randNum * (xTrainLength - model.batchsize))
  #bX, bY = x_train[randNum : randNum + model.batchsize], y_train[randNum : randNum + model.batchsize]
  
  
  #bX, bY = data.train.next_batch(model.batchsize)

  session.run(model.trainingstep, feed_dict={
    model.tx: bX, model.ty: bY, model.keep_prob: model.dropout
  })

  