from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import seaborn as sns 
from sklearn import metrics

__author__ = "Malte Carlstedt, Johan Östling"
# Both Malte and Johan spent about 4h in this part of the lab. 


# Loads the dataset from sklearn
iris = datasets.load_iris()
# Adding the names for the different flowers. Just later for labels.
class_names = iris.target_names

# Split the data into variables with 25% test set and 75% training set
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)


# Using the one-vs-rest option as requested in the assignment. Using this option since we have more than one input variable.
logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')

# Train the module by adjusting a line to fit our datapoints. 
logisticRegr.fit(x_train, y_train)

# Gives predicted flower from test values but does not return as numpy array
predictions = logisticRegr.predict(x_test)

# Gives a percentage of how many prediction was correct
score = logisticRegr.score(x_test, y_test)
print(score)

# Computing our confusion matrix to evaluate the accuracy of our predections contra the actual values.
cm = metrics.confusion_matrix(y_test, predictions)

# Using the library seaborn to plot the confusion matrix. The attributes are just settings for the matrix.
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')

# Posiitons of the tick marks in the matrix. So that we get setosa, versicolor, virginica on the correct posistion.
tick_marks = np.arange(0.5,len(class_names)+0.5)

# x-axis labels for the squares
plt.xticks(tick_marks, class_names)

# y-axis labels for the squares
plt.yticks(tick_marks, class_names)

# y-axis label for the actuals
plt.ylabel('True label')

# x-axis labels for the predicted
plt.xlabel('Predicted label')

# Setting the title of the plot
title = 'Accuracy Score: {0}'.format(score)
plt.title(title)

# Show the plot on the screen
plt.show()
