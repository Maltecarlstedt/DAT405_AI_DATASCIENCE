__author__ = "Malte Carlstedt, Johan Östling"

import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import collections
import matplotlib.pyplot as plt


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


  