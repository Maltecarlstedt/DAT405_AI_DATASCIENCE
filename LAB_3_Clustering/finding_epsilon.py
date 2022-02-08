from collections import Counter
from itertools import count
from matplotlib.colors import Colormap
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
data200 = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
X = data200.to_numpy()
#X = StandardScaler().fit_transform(X)

#Finding optimal value for epsilon
nearest_neighbors = NearestNeighbors(n_neighbors=4)
neighbors = nearest_neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]

#fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("Finding optimal Epsilon")

plt.show()