# Elbow algorithm collected from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
__author__ = "Malte Carlstedt, Johan Östling"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist


#save all data in file data_200.csv except the first row
allData = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del allData['residueName']
del allData['position']
del allData['chain']

# Convert to simple lists that is used to plot the clusters in the first assignment.
phi = allData['phi'].tolist()
psi = allData['psi'].tolist()

#turn data200 into a 2d numpy array
arr_2d = allData.to_numpy()

# Defineing lists to store our inertias and distorions.
distArray = []
inertiaArray = []

# For loop, testing for k-values between 1 to 15. 
K = range(1,15)
for k in K:
    
    # Using KMEans to build and then fit the model to our 2d array.
    km = KMeans(n_clusters=k).fit(arr_2d)
   
    # Adding all distortion value according to our distorion algorithm that is explained in the report
    distArray.append(sum(np.min(cdist(arr_2d, km.cluster_centers_,
                                        'euclidean'), axis=1)) / arr_2d.shape[0])
    # Adding inertias
    inertiaArray.append(km.inertia_)


#Printing for elbow method using Inertia
plt.plot(K, inertiaArray)
plt.xlabel('Values of K')
plt.ylabel('inertias')
plt.title('The Elbow Method using inertias')
plt.show()

"""
# Printing for elbow method using distorions
plt.plot(K, distArray)
plt.xlabel('Values of K')
plt.ylabel('distorions')
plt.title('The Elbow Method using Distortions')
plt.show()
"""




