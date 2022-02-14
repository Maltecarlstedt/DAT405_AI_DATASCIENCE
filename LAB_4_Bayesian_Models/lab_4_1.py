# Collected from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from os import listdir
from os.path import isfile, join

# Our paths. Need to use \\ otherwise Python interpets the path falsey.
hard_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\hard_ham"
easy_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\easy_ham"
spam_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\spam"

def readFiles(dir):
  fileContents = []
  # Get all the files in our directory
  fileNames = [f for f in listdir(dir) if isfile(join(dir, f))]
  # Open each file and read it's contents. Add to list.
  for i in range(len(fileNames)):
    # Need to use this otherwise the file path doesnt work.
    openFile = os.path.join(dir, fileNames[i])
    with open(openFile) as file:
      fileContents.append(file.read())
  return fileContents



easyTrain, easyTest = train_test_split(readFiles(easy_ham_path), test_size=0.3)


gaussianNB = GaussianNB()
