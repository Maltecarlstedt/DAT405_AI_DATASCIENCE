import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Read data from csv file
data = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data['residueName']
del data['position']
del data['chain']

#turn data into a 2d numpy array
X = data.to_numpy()


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