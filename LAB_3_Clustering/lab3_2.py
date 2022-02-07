from collections import Counter
from itertools import count
import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
X = data200.to_numpy()

"""
#Finding optimal value for epsilon
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(arr_2d)

distances, indices = neighbors.kneighbors(arr_2d)
distances = np.sort(distances[:,10], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")

plt.show()
"""
#Compute DBSCAN
X = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.4,min_samples=4).fit(X)
labels = db.labels_

print(labels)
#Number of clusters
n_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % n_clus)

#Identify outliers
n_noise = list(db.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)



for i in range(len(labels)-1):
    if labels[i]==-1:
        plt.scatter(X[i, 0], 
                    X[i, 1], 
                    c='black')
    else:
        plt.scatter(X[i, 0], 
                    X[i, 1], 
                    c=labels[i])

plt.show()

