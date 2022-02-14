# Collected from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

from audioop import mul
from fileinput import filename
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import datasets, metrics
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
import pandas as panda
import numpy

hard_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\hard_ham"
easy_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\easy_ham"
spam_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\spam"

# Our paths. Need to use \\ otherwise Python interpets the path falsey.
#hard_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\hard_ham"
#easy_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\easy_ham"
#spam_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\\LAB_4_Bayesian_Models\\spam"

# Read the contents of the file in the dir
def readFiles(dir):
  fileContents = []
  # Get all the files in our directory
  fileNames = [f for f in listdir(dir) if isfile(join(dir, f))]
  # Open each file and read it's contents. Add to list.
  for i in range(len(fileNames)):
    # Need to use this otherwise the file path doesnt work.
    openFile = os.path.join(dir, fileNames[i])
    # Latin 1 encoding needed to read spam-files.
    with open(openFile, encoding="Latin-1") as file:
      fileContents.append(file.read())
  return fileContents



#for every index there is a message
listOfHam = numpy.array(readFiles(easy_ham_path))
listOfSpam = numpy.array(readFiles(spam_path))

hamlabels = numpy.zeros((len(listOfHam),1))
spamlabels = numpy.ones((len(listOfSpam),1))


listOfHam = numpy.c_[listOfHam ,hamlabels]
listOfSpam = numpy.c_[listOfSpam,spamlabels]


#splits spam and ham mails to test and train sets
hamTrain, hamTest = train_test_split(listOfHam, test_size=0.3, random_state=0)
spamTrain, spamTest = train_test_split(listOfSpam, test_size=0.3, random_state=0)

#Setting up the variables which will build the model

x_train = numpy.concatenate((hamTrain[:,0], spamTrain[:,0]))
y_train = numpy.concatenate((hamTrain[:,1], spamTrain[:,1]))
x_test = numpy.concatenate((hamTest[:,0], spamTest[:,0]))
y_test = numpy.concatenate((hamTest[:,1], spamTest[:,1]))


# Vectorizing
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train_v = vectorizer.transform(x_train)
x_test_v = vectorizer.transform(x_test)

#Training and fitting the data with the multinomial naive bayes classifier
mnb_classifier = MultinomialNB()
mnb_classifier.fit(x_train_v, y_train)
mnb_predictions = mnb_classifier.predict(x_test_v)
print("Accuracy:",metrics.accuracy_score(y_test, mnb_predictions))
