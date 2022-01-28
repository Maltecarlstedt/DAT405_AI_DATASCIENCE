from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import seaborn as sns 
from sklearn import metrics




# Dataset from sklearn
iris = datasets.load_iris()
class_names = iris.target_names
print(class_names)

#split the data to 25% test set and 75% training set
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)


# Using the one vs rest option
logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')

# Train the module 
logisticRegr.fit(x_train, y_train)

#gives predicted flower from test values and returns as a numpy array
#logisticRegr.predict(x_test[0:len(x_test)-1]).reshape(1,-1)

#gives predicted flower from test values but does not return as numpy array
predictions = logisticRegr.predict(x_test)

#gives a percentage of how many prediction was correct
score = logisticRegr.score(x_test, y_test)
print(score)


cm = metrics.confusion_matrix(y_test, predictions)

#plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
tick_marks = np.arange(0.5,len(class_names)+0.5)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
# plt.savefig('toy_Digits_ConfusionSeabornCodementor.png') Saves it as a .png wtf lmao
plt.show()
