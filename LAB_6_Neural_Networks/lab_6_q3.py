from __future__ import print_function
from turtle import st
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np


# Parameters
batch_size = 128
num_classes = 10
epochs = 20
img_rows, img_cols = 28, 28

# Load the mnist dataset
(x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()

# Formatting
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Setting up
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.to_categorical(lbl_test, num_classes)

# Using 5 different Regozational factors between 0.000001 to 0.001
reg_factors = np.linspace(0.000001, 0.001, 5)

# List to keep track of our score from each iteration
score = []


# Defining Model
model = Sequential()
# Using Convulational layer Conv2d
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
# Using MaxPooling2D in between convulational layers. Pool Size of 2x2
model.add(MaxPooling2D(pool_size=(2,2)))
# Using Convulational layer Conv2d
model.add(Conv2D(64, kernel_size=(5, 5),activation='relu'))
# Using MaxPooling2D in between convulational layers. Pool Size of 2x2
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))

# Compiling Model
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.SGD(lr = 0.1),   
metrics=['accuracy'],)

fit_info = model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(x_test, y_test))
score.append(model.evaluate(x_test, y_test, verbose=0))

print(score)