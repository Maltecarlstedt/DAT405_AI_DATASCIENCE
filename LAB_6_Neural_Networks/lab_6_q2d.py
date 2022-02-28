from __future__ import print_function
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
import pandas as panda

#hyper-parameters data-loading and formatting
batch_size = 128
num_classes = 10
epochs = 40

# TODO GLÖM EJ KOMMENTERA OM ALLT 

img_rows, img_cols = 28, 28

(x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#preprocessing

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.to_categorical(lbl_test, num_classes)

reg_factors = np.linspace(0.000001, 0.001, 5)
score = []


for i in range(5):

  for j in range(3):
    ## Define model ##
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(500, activation = 'relu', kernel_regularizer=regularizers.l2(reg_factors[i])))
    model.add(Dense(300, activation = 'relu', kernel_regularizer=regularizers.l2(reg_factors[i])))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr = 0.1),
    metrics=['accuracy'],)
    
    fit_info = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
    score.append(model.evaluate(x_test, y_test, verbose=0))
    print("On stage:", i," and the", j, "th replication")
    print(score)

print(score)


panda.DataFrame(score, columns=["val_loss", "val_accuracy"]).to_csv("output_scores.csv")

