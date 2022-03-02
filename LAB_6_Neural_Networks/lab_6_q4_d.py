# This part of the lab is inspired by https://becominghuman.ai/the-deep-autoencoder-in-action-digit-reconstruction-bf177ccbb8c0
# Togheter with earlier questions code

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input

# Loading and splitting the mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatting our test and train data
x_train_flattened = X_train.reshape(60000, 784)
x_test_flattened = X_test.reshape(10000, 784)

# Setting up 
# Normalizing since pixels are represented as value between 0-255. We want only values between 0-1
x_train_flattened = x_train_flattened/255
x_test_flattened = x_test_flattened/255

# 'Deep' Auto encoder construction
input_image = Input(shape=(784,))
encoded_1 = Dense(32, activation='relu')(input_image)
laten_space_vector = Dense(2, activation='relu')(encoded_1)
decoded_1 = Dense(32, activation='relu')(laten_space_vector)
decoded_output = Dense(784, activation='sigmoid')(decoded_1)

# Auto Encoder
autoencoder = Model(inputs = input_image, outputs = decoded_output)
encoder_only = Model(inputs = input_image, outputs = laten_space_vector)

# Decoder
decoded_input = Input(shape=(2,))
decoder_layer = autoencoder.layers[-2](decoded_input)
decoded_output = autoencoder.layers[-1](decoder_layer)
decoder = Model(inputs = decoded_input, outputs = decoded_output)

# Compile with set optimizer and loss function.
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train (fitting) the model
autoencoder.fit(x_train_flattened, x_train_flattened, epochs=10, batch_size = 64, shuffle = True, validation_data=(x_test_flattened, x_test_flattened))

# Passes images along to our latent_space_vector.
encoded_pred = encoder_only.predict(x_train_flattened)

# Create image from our latent_space_vector
decoded_values_pred = decoder.predict(encoded_pred)


# After it has been in our decoder it has to be reshaped back into it's original shape.
decoded_values_pred = decoded_values_pred.reshape((60000, 28, 28))

# Just showing the original images in indexes 1 - 8
fig, axes = plt.subplots(ncols=8, figsize=(20, 7))
counter = 0
for i in range(1, 8):
    plt.title("Original images")
    axes[counter].set_title(y_train[i])
    axes[counter].imshow(X_train[i])
    counter += 1
plt.show()


# Showing the recreated images in indexes 1 - 8
fig, axes = plt.subplots(ncols=8, figsize=(20, 7))
counter = 0
for i in range(1,8 ):
    plt.title("Recreated images")
    axes[counter].set_title(y_train[i])
    axes[counter].imshow(decoded_values_pred[i])
    counter += 1
plt.show()