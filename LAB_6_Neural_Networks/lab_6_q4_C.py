# imports
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np




def salt_and_pepper(input, noise_level):
    # salt and pepper noise
    a = np.random.binomial(size=input.shape, n=1, p=(1 - noise_level))
    b = np.random.binomial(size=input.shape, n=1, p=0.5)
    c = (a==0) * b
    return input * a + c

# Hyper-parameters data-loading and formatting

batch_size = 128
num_classes = 10
epochs = 5

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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.np_utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(lbl_test, num_classes)

#data preparation

denoisedScore = []
#for denoised q2
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

    #following plotting code taken from https://blog.keras.io/building-autoencoders-in-keras.html
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs_test = encoder_only.predict(flattened_x_test_seasoneed)
    decoded_imgs_test = decoder.predict(encoded_imgs_test)

    encoded_imgs_train = encoder_only.predict(flattened_x_train_seasoned)
    decoded_imgs_train = decoder.predict(encoded_imgs_train)

    x_train_denoised = decoded_imgs_train.reshape(60000,28, 28)

    x_test_denoised = decoded_imgs_test.reshape(10000,28, 28)

    ## Define model ##
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr = 0.1),
            metrics=['accuracy'],)

    fit_info = model.fit(x_train_denoised, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test_denoised, y_test))
    score = model.evaluate(x_test_denoised, y_test, verbose=0)
    print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))
    denoisedScore.append(score[1])
print(denoisedScore)

"""
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(flattened_x_test_seasoneed[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print(i)
    plt.show()
"""
"""
#for seasoned images
seasonedScore = []
for i in [0, 0.2, 0.4, 0.6, 0.8, 1]:

    flattened_x_train = x_train.reshape(-1,784)
    flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=i)
    x_train_seasoned = flattened_x_train_seasoned.reshape(60000,28, 28)

    flattened_x_test = x_test.reshape(-1,784)
    flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=i)
    x_test_seasoned = flattened_x_test_seasoneed.reshape(10000,28, 28)

    ## Define model ##
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr = 0.1),
            metrics=['accuracy'],)

    fit_info = model.fit(x_train_seasoned, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test_seasoned, y_test))
    score = model.evaluate(x_test_seasoned, y_test, verbose=0)
    print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))
    seasonedScore.append(score[1])

print(seasonedScore)
"""