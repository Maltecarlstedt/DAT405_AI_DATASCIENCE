__author__ = "Malte Carlstedt, Johan Östling"
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read csv files.
data200 = pd.read_csv("data_200.csv", header = 0)
allData = pd.read_csv("data_all.csv", header = 0)
data500 = pd.read_csv("data_500.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del allData['residue name']
del allData['position']
del allData['chain']

# Convert to simple lists that is used to plot the clusters in the first assignment.
phi = allData['phi'].tolist()
psi = allData['psi'].tolist()

#turn data into a 2d numpy array
arr_2d = allData.to_numpy()


#taking the cosine function of every data point. This is for assignment 2D
#for i in range(len(arr_2d[:,0]-1)):
#    arr_2d[i,0]=math.cos(math.radians(arr_2d[i,0]))
#    arr_2d[i,1]=math.cos(math.radians(arr_2d[i,1]))


# Use KMeans to compute our clusters.
kmeans = KMeans(n_clusters=4, random_state=0)

#fits the centers to our clusters in our data set
kmeans.fit(arr_2d)

# Adding labels and titles
plt.title("Phi and psi combinations")
plt.ylabel("psi")
plt.xlabel("phi")

#x = phi, y = psi, Using c=kmeans.labels to colour them.
plt.scatter(arr_2d[:,0],arr_2d[:,1], c=kmeans.labels_.astype(float))

#plots the centers for each cluster.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x',s=150, linewidths = 3)
plt.show()

#plotting without colouring and centers. For assignment 1
#plt.scatter(phi,psi)
#plt.show()
