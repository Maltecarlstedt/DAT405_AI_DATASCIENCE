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



# Read all files in a directory. Convert into a numpy array and add to list.
#listOfHam = numpy.array(readFiles(easy_ham_path)) # Uncomment for easy ham
listOfHam = numpy.array(readFiles(hard_ham_path)) # Uncomment for hard ham
listOfSpam = numpy.array(readFiles(spam_path))

#flatlistHam = [word for email in listOfHam for word in email.split(" ")]  # A flat list of all words from all emails

flatlistSpam = [word for email in listOfSpam for word in email.split(" ")]  # A flat list of all words

word_count = collections.Counter(flatlistSpam) #counts number of apperases of every word in the mails

mostCommonWords = []
numOfApperanses = []

for word, count in word_count.most_common():
    mostCommonWords.append(word)
    numOfApperanses.append(count)

# Removing whitespace    
mostCommonWords.pop(0)
numOfApperanses.pop(0)
#c.most_common()[:-n-1:-1]   
fig = plt.figure()
plt.bar(list(mostCommonWords),list(numOfApperanses))
plt.xlabel("words")
plt.ylabel("Number of apperanses")
plt.title("Top 15 most common words in all spam emails")
plt.show()
