from __future__ import print_function
from turtle import st
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import numpy as np
from keras import regularizers
import pandas as panda


# Parameters
batch_size = 128
num_classes = 10
epochs = 40
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

# These loops take about 20min to run. The results can be seen in the report.
# First loop for running each factor
for i in range(5):
    # Second loop for running each factor three times.
  for j in range(3):
    # Defining Model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(500, activation = 'relu', kernel_regularizer=regularizers.l2(reg_factors[i])))
    model.add(Dense(300, activation = 'relu', kernel_regularizer=regularizers.l2(reg_factors[i])))
    model.add(Dense(num_classes, activation='softmax'))

    # Compiling Model
    model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr = 0.1),
    metrics=['accuracy'],)
    
    fit_info = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
    # Adding score to list
    score.append(model.evaluate(x_test, y_test, verbose=0))
    # Just som printing for each iteration to see progress.
    print("On stage:", i," and the", j, "th replication")
    print(score)

print(score)

# Using pandas to store accurcy from each iteration. 
panda.DataFrame(score, columns=["val_loss", "val_accuracy"]).to_csv("output_scores.csv")

# This next part was done after the above loops were finished since the calculation took so long time.
# Reading from the csv file that we created with the output from the above loops.
df = panda.read_csv("output_scores.csv")
# Extracting val_accuracy column.
val_accuracy = np.array(df["val_accuracy"])


means = [] #five means for each regularizers
standardD = [] #five standard deviations for each regularizers

# Since our results is one big list with every iteration we need to extract them with i = i+3.
for i in range(5):
    accuracy = np.array([val_accuracy[i], val_accuracy[i+1 ],val_accuracy[i+2]])
    means.append(accuracy.mean())
    standardD.append(accuracy.std())
    i=i+3

# Debugging.
print(means)
print(standardD)