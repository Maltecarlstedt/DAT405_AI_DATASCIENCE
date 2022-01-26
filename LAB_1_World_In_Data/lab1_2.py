import pandas as pd 
import matplotlib.pyplot as scatterPlot
import csv

# Reading from CSV-files
hapiness = pd.read_csv("hapiness.csv", header = 0)
trust = pd.read_csv("trust.csv", header = 0)
corruption = pd.read_csv("corruption.csv", header = 0)

# Data cleaning, only using data from a specific year
# Using 2014 here since this the most recent year the data has in common
hapiness = hapiness[hapiness['Year']==2014]
trust = trust[trust['Year']==2014]
corruption = corruption[corruption['Year']==2014]

# Data cleaning the column "code" sine it has no use to us.
del hapiness["Code"]
del trust["Code"]
del corruption["Code"]

# To get lists from CSV-files
hapinessCountries = hapiness['Entity'].tolist()
trustCountries = trust['Entity'].tolist()
corruptionCountries = corruption['Entity'].tolist()

# To get the countries that exists in all 3 lists.
def getMatchingCountries(listA, listB, listC):
 matchingCountries = list(set(listA) & set(listB) & set(listC))
 return matchingCountries

# Creating a list of all countries that exists in all files
matchingCountries = getMatchingCountries(corruptionCountries, hapinessCountries, trustCountries)

# Sorting it so it's in the same order as the rows in the csv-files.
matchingCountries.sort()

# Removing rows that are not in matchingCountries
hapiness = hapiness.loc[hapiness['Entity'].isin(matchingCountries)]
trust = trust.loc[trust['Entity'].isin(matchingCountries)]
corruption = corruption.loc[corruption['Entity'].isin(matchingCountries)]

# Debugging
#hapinessCountries = hapiness['Entity'].tolist()
#trustCountries = trust['Entity'].tolist()

# Creating list of our data columns
hapinessData = hapiness['data'].tolist()
corruptionData = corruption['data'].tolist()
trustData = trust['data'].tolist()

# Plotting
scatterPlot.scatter(corruptionData , hapinessData)
#For labeling the data
for i in range(len(matchingCountries)-1):
  scatterPlot.annotate(matchingCountries[i], (corruptionData[i], hapinessData[i])) #for us to see which point represents the country
scatterPlot.ylabel("Happiness - higher is more happy")
scatterPlot.xlabel("Corruption - lower is more corrupt")
scatterPlot.show()

# Labels
# Corruption Perception Index - Transparency International (2018)
# Life satisfaction in Cantril Ladder (World Happiness Report 2021)
# Trust in others - percentages of people agreeing with the statement "I can trust others"