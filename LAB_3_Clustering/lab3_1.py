import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#delete columns we dont want
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
arr_2d = data200.to_numpy()

#x = phi, y = psi
plt.scatter(arr_2d[:,0],arr_2d[:,1])

#from the scatter plot we see that the points form 4 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

#fits the centers to 4 clusters in our data set
kmeans.fit(arr_2d)

#plots the centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x',s=150, linewidths = 3)
plt.show()

"""
#plotting and adding titles
plt.title("Phi and psi combinations")
plt.ylabel("psi")
plt.xlabel("phi")
plt.scatter(phi,psi)
plt.show()
"""