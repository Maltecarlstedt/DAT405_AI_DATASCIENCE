__author__ = "Malte Carlstedt, Johan Östling"
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Read data from csv file
data = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data['residueName']
del data['position']
del data['chain']

#turn data into a 2d numpy array
X = data.to_numpy()

# Using NearestNeighbors to find how many has n_neighbours of a certain value.
closestNeighbours = NearestNeighbors(n_neighbors=60)
# Fits our 2d array accordingly
neighbours = closestNeighbours.fit(X)
# Calculating distance and indices
distance, indices = neighbours.kneighbors(X)
# Sort the distances in order.
distance = np.sort(distance, axis=0)
# Using distance
distance = distance[:,1]

# Plotting
plt.plot(distance)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("Finding optimal Epsilon")

plt.show()