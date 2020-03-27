import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model as model
import random as rand
import numpy as np


def encoding(x):
  arr = np.zeros((10, 1))
  arr[x] = 1
  return np.reshape(arr, (10,));

# getting the data set from tensorflow
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

y_train = np.array([encoding(x) for x in y_train])

y_test = np.array([encoding(x) for x in y_test])

x_train = np.array([np.reshape(x, (784,)) for x in x_train])

x_test = np.array([np.reshape(x, (784,)) for x in x_test])

print(y_train[0])

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
  randNum = rand.random()
  randNum = int(randNum * (xTrainLength - model.batchsize))
  bX, bY = x_train[randNum : randNum + model.batchsize], y_train[randNum : randNum + model.batchsize]
  
  

  session.run(model.trainingstep, feed_dict={
    model.tx: bX, model.ty: bY, model.keep_prob: model.dropout
  })



# printing training accuracy
loss, a = session.run([lossfunction, accuracy], feed_dict={
    tx: bX, ty: bY, keep_prob: 1
})

print(loss)
print(a)


# testing accuracy
testAccuracy = session.run(accuracy, feed_dict={
    tx: x_test, ty: y_test, keep_prob: 1
})

print(testAccuracy)


img0 = np.invert(Image.open("0.png").convert('L')).ravel()
predict0 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img0], keep_prob: 1})
print ("Prediction for 0 image:", np.squeeze(predict0))

img1 = np.invert(Image.open("1.png").convert('L')).ravel()
predict1 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img1], keep_prob: 1})
print ("Prediction for 1 image:", np.squeeze(predict1))

img2 = np.invert(Image.open("2.png").convert('L')).ravel()
predict2 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img2], keep_prob: 1})
print ("Prediction for 2 image:", np.squeeze(predict2))

img3 = np.invert(Image.open("3.png").convert('L')).ravel()
predict3 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img3], keep_prob: 1})
print ("Prediction for 3 image:", np.squeeze(predict3))

img4 = np.invert(Image.open("4.png").convert('L')).ravel()
predict4 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img4], keep_prob: 1})
print ("Prediction for 4 image:", np.squeeze(predict4))

img5 = np.invert(Image.open("5.png").convert('L')).ravel()
predict5 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img5], keep_prob: 1})
print ("Prediction for 5 image:", np.squeeze(predict5))

img6 = np.invert(Image.open("6.png").convert('L')).ravel()
predict6 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img6], keep_prob: 1})
print ("Prediction for 6 image:", np.squeeze(predict6))

img7 = np.invert(Image.open("7.png").convert('L')).ravel()
predict7 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img7], keep_prob: 1})
print ("Prediction for 7 image:", np.squeeze(predict7))

img8 = np.invert(Image.open("8.png").convert('L')).ravel()
predict8 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img8], keep_prob: 1})
print ("Prediction for 8 image:", np.squeeze(predict8))

img9 = np.invert(Image.open("9.png").convert('L')).ravel()
predict9 = session.run(tf.argmax(outputLayer, 1), feed_dict={tx: [img9], keep_prob: 1})
print ("Prediction for 9 image:", np.squeeze(predict9))

