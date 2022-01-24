import pandas as pd 
import matplotlib.pyplot as scatterPlot
import csv

# Reading from CSV-files
hapiness = pd.read_csv("hapiness.csv", header = 0)
higherEducation = pd.read_csv("higherEducation.csv", header = 0)
trust = pd.read_csv("trust.csv", header = 0)

# Data cleaning, only using data from a specific year
hapiness = hapiness[hapiness['Year']==2010]
higherEducation = higherEducation[higherEducation['Year']==2010]
# Using 2009 here since no other year was good four us.
trust = trust[trust['Year']==2009]

# Data cleaning the column "code" sine it has no use to us.
del hapiness["Code"]
del higherEducation["Code"]
del trust["Code"]

# To get lists from CSV-files
higherEducationCountries = higherEducation['Entity'].tolist()
hapinessCountries = hapiness['Entity'].tolist()
trustCountries = trust['Entity'].tolist()

# To get the countries that exists in all 3 lists.
def getMatchingCountries(listA, listB, listC):
 matchingCountries = list(set(listA) & set(listB) & set(listC))
 return matchingCountries

# Creating a list of all countries that exists in all files
matchingCountries = getMatchingCountries(higherEducationCountries, hapinessCountries, trustCountries)

# Removing rows that are not in matchingCountries
hapiness = hapiness.loc[hapiness['Entity'].isin(matchingCountries)]
trust = trust.loc[trust['Entity'].isin(matchingCountries)]
higherEducation = higherEducation.loc[higherEducation['Entity'].isin(matchingCountries)]

# Debugging
#higherEducationCountries = higherEducation['Entity'].tolist()
#hapinessCountries = hapiness['Entity'].tolist()
#trustCountries = trust['Entity'].tolist()

# Creating list of our data columns
higherEducationData = higherEducation['data'].tolist()
hapinessData = hapiness['data'].tolist()
trustData = trust['data'].tolist()

# Plotting
scatterPlot.scatter(trustData, hapinessData)
scatterPlot.show()