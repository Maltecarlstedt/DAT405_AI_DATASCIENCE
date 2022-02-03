from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

__author__ = "Malte Carlstedt, Johan Östling"
# Both Malte and Johan spent about 2h on this part of the lab

# Loads the dataset from sklearn
iris = datasets.load_iris()

# Adding the names for the different flowers. Just later for labels.
class_names = iris.target_names

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# Creates the model. Where k = number of neighbours.
k= 112
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the model with our training set.
knn.fit(x_train, y_train)

# Predicts the y value using the x_test value. 
y_pred = knn.predict(x_test)

# Gives a percentage of how many prediction was correct
score = metrics.accuracy_score(y_test, y_pred)

# Computing our confusion matrix to evaluate the accuracy of our predections contra the actual values.
cm = metrics.confusion_matrix(y_test, y_pred)

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
title = title + ' , for k = '+str(k)
plt.title(title)

# Show the plot on the screen
plt.show()
