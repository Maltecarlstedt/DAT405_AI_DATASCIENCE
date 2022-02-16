# Read file function inspired from stackoverflow: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
__author__ = "Malte Carlstedt, Johan Östling"

# Below follow code related to question 4 in the assignment.
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy
import collections



# For Johans Computer
hard_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\hard_ham"
easy_ham_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\easy_ham"
spam_path = "C:\\Users\johan\OneDrive\Dokument\Introduction to data science and AI\DAT405_AI_DATASCIENCE\LAB_4_Bayesian_Models\spam"

# For Maltes Computer
# Our paths. Need to use \\ otherwise Python interpets the path falsey.
#hard_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\hard_ham"
#easy_ham_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\LAB_4_Bayesian_Models\\easy_ham"
#spam_path = "C:\\Users\Malte Carlstedt\\DAT405_AI_DS\\LAB_4_Bayesian_Models\\spam"

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

wordsHam = [word for email in listOfHam for word in email.split(" ")]  #list of all words from ham

wordsSpam = [word for email in listOfSpam for word in email.split(" ")]  #list of all words from spam

word_count_spam = collections.Counter(wordsSpam) #counts number of appearances of every word in spam mails
word_count_ham = collections.Counter(wordsHam) #counts number of appearances of every word in ham mails. 

#to make a bar chart we add the most common words to one list and the number of appearances to another. 
mostCommonWordsHam = []
numOfApperansesHam = []

mostCommonWordsSpam = []
numOfApperansesSpam = []

#for the most uncommmon words we are not interested in doing a bar chart, therefore the number of appearances is not relevant
mostUnCommonWordsHam = []
mostUnCommonWordsSpam = []

#returns top 16 most common words
for word, count in word_count_spam.most_common(16):
    mostCommonWordsSpam.append(word)
    numOfApperansesSpam.append(count)

for word, count in word_count_ham.most_common(16):
    mostCommonWordsHam.append(word)
    numOfApperansesHam.append(count)

#return words that only occur once
for word, count in word_count_spam.items():
    if count == 1:
      mostUnCommonWordsSpam.append(word)

for word, count in word_count_ham.items():
    if count == 1:
      mostUnCommonWordsHam.append(word)


# Removing whitespace    
mostCommonWordsHam.pop(0)
numOfApperansesHam.pop(0)

mostCommonWordsSpam.pop(0)
numOfApperansesSpam.pop(0)

#returning a list with the intersection of the most common words in spam and ham. 
matchingWords = list(set(mostCommonWordsHam) & set(mostCommonWordsSpam))


fig = plt.figure()
# Switch to mostCommonWordsSpam/numOfApperansesSpam when needed.
plt.bar(list(mostCommonWordsHam),list(numOfApperansesHam))
plt.xlabel("words")
plt.ylabel("Number of apperanses")
plt.title("Top 15 most common words in all spam emails")
plt.show()
