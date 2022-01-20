import pandas as pd 
import matplotlib.pyplot as scatterPlot

gdpCountries = pd.read_csv("LAB_1_World_In_Data\gdp.csv", header = 0)
lifeExpecCountries = pd.read_csv("LAB_1_World_In_Data\life_expec.csv", header = 0)

matchingCountries = []
lifeExpec = []
gdp = []

i = 0
j = 0

# Slow code :( 
for index, gdpRows in gdpCountries.iterrows():
  for index, lifeRows in lifeExpecCountries.iterrows():
    if(gdpRows[0] == lifeRows[0]):
      matchingCountries.append(lifeRows[0])
      if(gdpRows[0] == matchingCountries[i]):
        gdp.append(gdpRows[3])
        i = i + 1
      if(lifeRows[0] == matchingCountries[j]):
        lifeExpec.append(lifeRows[3])
        j = j + 1

scatterPlot.scatter(gdp, lifeExpec)
scatterPlot.show()


""""
print("_____________________")
print(matchingCountries)
print("_____________________")
print(lifeExpec)
print("_____________________")
print(gdp)
print("_____________________")
"""






      

