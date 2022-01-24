import pandas as pd 
import matplotlib.pyplot as scatterPlot
import math

hapiness = pd.read_csv("hapiness.csv", header = 0)
higherEducation = pd.read_csv("higherEducation.csv", header = 0)

# Data cleaning, only using the most recent data.
hapiness = hapiness[hapiness['Year']==2010]
higherEducation = higherEducation[higherEducation['Year']==2010]

# Data cleaning the column "code" sine it has no use to us.
del hapiness["Code"]
del higherEducation["Code"]



