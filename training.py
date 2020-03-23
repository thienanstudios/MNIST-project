import tensorflow as tf
import model as model

# getting the data set from tensorflow
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

print(y_train)


# normalize dataset
# max value is 255 because 255 is the rgb value
(float)x_train = (float)x_train / 255
(float)x_test = (float)x_test / 255





