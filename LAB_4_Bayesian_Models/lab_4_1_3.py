# Read file function inspired from stackoverflow: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

__author__ = "Malte Carlstedt, Johan Östling"

import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
import numpy


# For Johans Computer
#hard_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\hard_ham"
#easy_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\easy_ham"
#spam_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\spam"

# For Maltes Computer
# Our paths. Need to use \\ otherwise Python interpets the path falsey.
hard_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\hard_ham"
easy_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\easy_ham"
spam_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\\LAB_4_Bayesian_Models\\spam"

# Read the contents of the file in the dir
def readFiles(dir):
  fileContents = []
  # Get all the files (filenames) in our directory
  fileNames = [f for f in listdir(dir) if isfile(join(dir, f))]
  # Open each file and read it's contents. Add to list.
  for i in range(len(fileNames)):
    # Need to use os.path.join otherwise the file path gets incorrect. Ordinary string concatenate doesn't work.
    openFile = os.path.join(dir, fileNames[i])
    # Latin 1 encoding needed to read spam-files.
    with open(openFile, encoding="Latin-1") as file:
      # Read each file and add to list.
      fileContents.append(file.read())
  return fileContents


# Read all files in a directory. Convert into a numpy array and add to list.
#listOfHam = numpy.array(readFiles(easy_ham_path)) # Uncomment for easy ham
listOfHam = numpy.array(readFiles(hard_ham_path)) # Uncomment for hard ham
listOfSpam = numpy.array(readFiles(spam_path))

# Creates a list with zeros of length listOfHam
hamlabels = numpy.zeros((len(listOfHam),1))
# Creates a list with ones of length listOfSpam
spamlabels = numpy.ones((len(listOfSpam),1))

# Add the labels as columns. Need to add labels to be able to differentiate spam and no spam.
# Label 1 = spam, Label 0 = no spam.
listOfHam = numpy.c_[listOfHam ,hamlabels]
listOfSpam = numpy.c_[listOfSpam,spamlabels]


# Splits spam and ham data to test and train sets. using random_state=0 to get same result each time.
hamTrain, hamTest = train_test_split(listOfHam, test_size=0.3, random_state=0)
spamTrain, spamTest = train_test_split(listOfSpam, test_size=0.3, random_state=0)


# Setting up variables that the model will use. Choosing data by specifying each column.
# X are the mails. Y is the label.
# Using numpy.concatenate to join the lists.
x_train = numpy.concatenate((hamTrain[:,0], spamTrain[:,0]))
y_train = numpy.concatenate((hamTrain[:,1], spamTrain[:,1]))
x_test = numpy.concatenate((hamTest[:,0], spamTest[:,0]))
y_test = numpy.concatenate((hamTest[:,1], spamTest[:,1]))


# Using CountVectorizer to transform our emails into vectors to be able to classify text.
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
trained_x_vector = vectorizer.transform(x_train)
x_test_vector = vectorizer.transform(x_test)

"""
# Using Naïve Baye multinomial classifier to train our datasets.
multiNB = MultinomialNB()
multiNB.fit(trained_x_vector, y_train)
multiNB_predict = multiNB.predict(x_test_vector)
print("Accuracy Multinomial:",metrics.accuracy_score(y_test, multiNB_predict))

# Using Naïve Baye bernoulli classifier to train our datasets.
bernoulliNB = BernoulliNB(binarize=1.0)
bernoulliNB.fit(trained_x_vector,y_train)
bernoulliNB_predict = bernoulliNB.predict(x_test_vector)
print("Accuracy Bernoulli:",metrics.accuracy_score(y_test, bernoulliNB_predict))
"""

