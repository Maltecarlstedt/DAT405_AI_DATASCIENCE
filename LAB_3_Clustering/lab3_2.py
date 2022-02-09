__author__ = "Malte Carlstedt, Johan Östling"
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv file
data = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data['residueName']
del data['position']
del data['chain']

#turn data into a 2d numpy array
X = data.to_numpy()

# Using DBSCAN to compute our data
db = DBSCAN(eps=12, metric="euclidean", min_samples=70).fit(X)

# Find number of clusters by checking their given labels and allowing no duplicates => gives clusters.
numberOfClusters =len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
print('Estimated no. of clusters: ', numberOfClusters)
# Find outliers through which label they were given.
numberOfNoisePoints = list(db.labels_).count(-1)
print('Estimated number of of noise points: ', numberOfNoisePoints)

# Scatter our 2d array by column 0 and 1. Colouring according to their labels.
plt.scatter(X[:, 0], 
            X[:, 1], 
            c=db.labels_)


# Scatter over our plots that are considered as outliers by checking their label. Probably not the best complexity but it works. Colouring all outliers black.
for i in range(len(db.labels_)-1):
    if db.labels_[i]==-1:
        plt.scatter(X[i, 0], 
                    X[i, 1], 
                    c='black')



plt.title("Min_samples = 70 and Epsilon = 12")
plt.show()
