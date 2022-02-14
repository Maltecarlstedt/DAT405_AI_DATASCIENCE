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



# Our paths. Need to use \\ otherwise Python interpets the path falsey.
hard_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\hard_ham"
easy_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\easy_ham"
spam_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\\LAB_4_Bayesian_Models\\spam"

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



listOfHam = readFiles(easy_ham_path)
listOfSpam = readFiles(spam_path)

hamTrain, hamTest = train_test_split(listOfHam, test_size=0.3, random_state=0)
spamTrain, spamTest = train_test_split(listOfSpam, test_size=0.3, random_state=0)

# Merging trainingsets 
trainingSetX = hamTrain + spamTrain
trainingSetY = spamTrain + hamTrain
# Merging test sets
testSet = hamTest + spamTest


vectorizer = CountVectorizer()
vectorizer.fit(trainingSetX)
training_set_vec = vectorizer.transform(trainingSetX)
test_set_vec = vectorizer.transform(testSetY)

multiNomNb = MultinomialNB()
multiNomNb.fit(training_set_vec,  )