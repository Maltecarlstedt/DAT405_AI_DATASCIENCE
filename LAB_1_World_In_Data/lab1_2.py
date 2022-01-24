import pandas as pd 
import matplotlib.pyplot as scatterPlot
import csv


hapiness = pd.read_csv("hapiness.csv", header = 0)
higherEducation = pd.read_csv("higherEducation.csv", header = 0)
savings = pd.read_csv("savings.csv", header = 0)

# Data cleaning, only using the most recent data.
hapiness = hapiness[hapiness['Year']==2010]
higherEducation = higherEducation[higherEducation['Year']==2010]
savings = savings[savings['Year']==2010]

# Data cleaning the column "code" sine it has no use to us.
del hapiness["Code"]
del higherEducation["Code"]
del savings["Code"]

hapinessCountries = []
higherEducationCountries = []
savingsCountries = []

# To get lists from CSV-files
higherEducationCountries = higherEducation['Entity'].tolist()
hapinessCountries = hapiness['Entity'].tolist()
savingsCountries = savings['Entity'].tolist()

#higherEducationData = higherEducation['higherED'].tolist()
#hapinessData = hapiness['hapiness'].tolist()
#savingsData = savings['savings'].tolist()


# To get the countries that exists in 2 lists.
def getMatchingCountries(listA, listB, listC):
 matchingCountries = list(set(listA) & set(listB) & set(listC))
 return matchingCountries

matchingCountries = getMatchingCountries(higherEducationCountries, hapinessCountries, savingsCountries)



"""""
def get_Correlated_data(listA, listB):
  matchingCountries = getMatchingCountries(listA,listB)
  for i in range(len(listA)):
    for j in range(len(listB)):
      if
"""
#scatterPlot.scatter(hapinessData, higherEducationData)
#scatterPlot.show()