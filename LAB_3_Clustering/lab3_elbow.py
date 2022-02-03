# Elbow algorithm collected from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist


#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
arr_2d = data200.to_numpy()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
 
K = range(1,15)
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

# Printing for elbow method using distorions
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('distorions')
plt.title('The Elbow Method using Distortions')
plt.show()

# Printing for elbow method using distorions
#plt.plot(K, inertia, 'bx-')
#plt.xlabel('Values of K')
#plt.ylabel('inertias')
#plt.title('The Elbow Method using inertias')
#plt.show()

