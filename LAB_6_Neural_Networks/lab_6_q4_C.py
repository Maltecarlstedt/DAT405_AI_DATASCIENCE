from __future__ import print_function
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

from LAB_6_Neural_Networks.lab_6_q4 import salt_and_pepper

#hyper-parameters data-loading and formatting

batch_size = 128
num_classes = 10
epochs = 10

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

for i in [0, 0.2, 0.4, 0.6, 0.8, 1]:
  flattened_x_train = x_train.reshape(-1,784)
  flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=i)

  flattened_x_test = x_test.reshape(-1,784)
  flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=i)



  seasonedScore = []
  ## Define model ##
  model = Sequential()

  model.add(Flatten())
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(num_classes, activation='softmax'))


  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr = 0.1),
          metrics=['accuracy'],)

  fit_info = model.fit(flattened_x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(flattened_x_test, y_test))
  seasonedScore = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss: {}, Test accuracy {}'.format(seasonedScore[0], seasonedScore[1]))


for i in [0, 0.2, 0.4, 0.6, 0.8, 1]:
  flattened_x_train = x_train.reshape(-1,784)
  flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=i)

  flattened_x_test = x_test.reshape(-1,784)
  flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=i)

  latent_dim = 96  

  input_image = keras.Input(shape=(784,))
  encoded = Dense(128, activation='relu')(input_image)
  encoded = Dense(latent_dim, activation='relu')(encoded)
  decoded = Dense(128, activation='relu')(encoded)
  decoded = Dense(784, activation='sigmoid')(decoded)

  autoencoder = keras.Model(input_image, decoded)
  encoder_only = keras.Model(input_image, encoded)

  encoded_input = keras.Input(shape=(latent_dim,))
  decoder_layer = Sequential(autoencoder.layers[-2:])
  decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

  fit_info_AE = autoencoder.fit(flattened_x_train_seasoned, flattened_x_train,
                  epochs=32,
                  batch_size=64,
                  shuffle=True,
                  validation_data=(flattened_x_test_seasoneed, flattened_x_test))


  encoded_imgs = encoder_only.predict(flattened_x_test_seasoneed)
  decoded_imgs = decoder.predict(encoded_imgs)

  denoisedScore = []

  ## Define model ##
  model = Sequential()

  model.add(Flatten())
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(num_classes, activation='softmax'))


  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr = 0.1),
          metrics=['accuracy'],)

  fit_info = model.fit(flattened_x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(flattened_x_test, y_test))
  seasonedScore = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss: {}, Test accuracy {}'.format(seasonedScore[0], seasonedScore[1]))

