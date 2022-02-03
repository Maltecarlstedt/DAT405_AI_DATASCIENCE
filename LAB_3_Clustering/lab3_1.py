import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#delete columns we dont want
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
arr_2d = data200.to_numpy()

K = range(1, 10)
 
for k in K:
  clusterer = KMeans(n_clusters=k, random_state=0)
  cluster_labels = clusterer.fit_predict(arr_2d)

  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(arr_2d, cluster_labels)
  print(
        "For n_clusters =",
        k,
        "The average silhouette_score is :",
        silhouette_avg,
    )



"""
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 15)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(arr_2d)
    kmeanModel.fit(arr_2d)
 
    distortions.append(sum(np.min(cdist(arr_2d, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / arr_2d.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(arr_2d, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / arr_2d.shape[0]
    mapping2[k] = kmeanModel.inertia_


plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertias')
plt.title('The Elbow Method using Inertias')
plt.show()


_______________________________________________________________-
#from the scatter plot we see that the points form 4 clusters
kmeans = KMeans(n_clusters=5, random_state=0)

#fits the centers to 4 clusters in our data set
kmeans.fit(arr_2d)

# Adding labels and titles
plt.title("Phi and psi combinations")
plt.ylabel("psi")
plt.xlabel("phi")

#x = phi, y = psi, Using c=kmeans.labels to colour them.
plt.scatter(arr_2d[:,0],arr_2d[:,1], c=kmeans.labels_.astype(float))

#plots the centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x',s=150, linewidths = 3)
plt.show()


#plotting
plt.scatter(phi,psi)
plt.show()
"""

