import matplotlib.pyplot as scatterPlot
import csv
import numpy as np
from csv import reader

#gdpFile = open("gdp (1).csv")
#lifeFile = open("life_expec.csv")

#csvreaderGdp = csv.reader(gdpFile)
#csvreaderLife = csv.reader(lifeFile)

countryLife = []
countryGdp = []
finalCountry = [] #to get list of every country both gpa and life expectancy have in common
lifeExpectancy = []
gdpCapita = []

with open('life_expec.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader: 
      countryLife.append(row[0])

with open('gdp.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader: 
        countryGdp.append(row[0])

countryLife_as_set = set(countryLife)
intersection = countryLife_as_set.intersection(countryGdp)

finalCountry = list(intersection)

i=0

while i<(len(finalCountry)-1):
    with open('life_expec.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader: 
            if row[0]==finalCountry[i]:
                lifeExpectancy.append(row[3])
                continue
    with open('gdp.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader: 
            if row[0]==finalCountry[i]:
                gdpCapita.append(row[3])
                continue
    i=i+1




scatterPlot.scatter(gdpCapita,lifeExpectancy) 
scatterPlot.show()
