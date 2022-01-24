from numpy import average
import pandas as pd 
import matplotlib.pyplot as scatterPlot
import math

# importing from CSV files. Note the files has already been data cleaned and only has data from 2018. 
gdpCountries = pd.read_csv("gdp.csv", header = 0)
lifeExpecCountries = pd.read_csv("life_expec.csv", header = 0)

# Data cleaning the column "code" sine it has no use to us.
del gdpCountries["Code"]
del lifeExpecCountries["Code"]

# print(gdpCountries.to_string)
# print(lifeExpecCountries.to_string)
# Initiating list that are going to be used
matchingCountries = []
lifeExpec = []
gdp = []

# Used for index
i = 0
j = 0

# Slow code, there are better Pandas method for iterating over lists. 
# This is the method that lets us find the countries that are both from the gdp csv file and the life expectancy file
for index, gdpRows in gdpCountries.iterrows():
  for index, lifeRows in lifeExpecCountries.iterrows():
    if(gdpRows[0] == lifeRows[0]):
      matchingCountries.append(lifeRows[0])
      if(gdpRows[0] == matchingCountries[i]):
        gdp.append(gdpRows[2])
        i = i + 1
      if(lifeRows[0] == matchingCountries[j]):
        lifeExpec.append(lifeRows[2])
        j = j + 1

# To calculate the average for a list.
def Average(list):
  sum = 0
  for i in range(len(list)-1):
    sum=sum+list[i]
  return sum/len(list)

# To calculate the standard deviation for a list
def Standard_dev(list):
  average = Average(list)
  deviations = []
  for i in range(len(list)-1):
    deviations.append((list[i]-average)**2)
  almostdone = Average(deviations)
  return math.sqrt(almostdone)

# Debugging
# print(Average(lifeExpec))
# print(Standard_dev(lifeExpec))
# print(Average(gdp))
# print(Standard_dev(gdp))

# Returns a list of all countries that has a lower gdp than average.
def Countries_below_mean(list):
  average = Average(list)
  below_mean = []
  for index, gdpRows in gdpCountries.iterrows():
    if(gdpRows[3] < average and gdpRows[0] in matchingCountries):
      below_mean.append(gdpRows[0])
  return below_mean

# Returns a list of all countries that has a higher gdp than average
def Countries_above_mean(list):
  average = Average(list)
  below_mean = []
  for index, gdpRows in gdpCountries.iterrows():
    if(gdpRows[3] > average and gdpRows[0] in matchingCountries):
      below_mean.append(gdpRows[0])
  return below_mean
  
# Returns a list of all countries with one standard deviation above the average.
def Countries_above_standard(list):
  oneAboveStandardDev = Standard_dev(list) + Average(list)
  listOfCountriesAbove = []
  for index, lifeRows in lifeExpecCountries.iterrows():
    if(lifeRows[2] >= oneAboveStandardDev and lifeRows[0] in matchingCountries):
      listOfCountriesAbove.append(lifeRows[0])
  return listOfCountriesAbove

# Returns a list of all countries with one standard deviation below the average
def Countries_below_standard(list):
  oneAboveStandardDev = -Standard_dev(list) + Average(list)
  listOfCountriesAbove = []
  for index, lifeRows in lifeExpecCountries.iterrows():
    if(lifeRows[3] <= oneAboveStandardDev and lifeRows[0] in matchingCountries):
      listOfCountriesAbove.append(lifeRows[0])
  return listOfCountriesAbove


# To get the matching countries in every list. 
# print(list(set(Countries_below_standard(lifeExpec)) & set(Countries_above_mean(gdp))))
# print(list(set(Countries_above_standard(lifeExpec)) & set(Countries_below_mean(gdp))))


# Scatter our plot. The for loop is optional and is used to put labels on each data point.
scatterPlot.scatter(gdp, lifeExpec)
for i in range(len(matchingCountries)-1):
  scatterPlot.annotate(matchingCountries[i], (gdp[i], lifeExpec[i])) #for us to see which point represents the country
scatterPlot.ylabel("life-expectancy")
scatterPlot.xlabel("gdp per capita")
scatterPlot.show()

# Debugging
""""
print("_____________________")
print(matchingCountries)
print("_____________________")
print(lifeExpec)
print("_____________________")
print(gdp)
print("_____________________")
"""






      

