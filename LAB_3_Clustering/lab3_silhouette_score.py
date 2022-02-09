__author__ = "Malte Carlstedt, Johan Östling"
import math
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Read data from Csv-file
allData = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del allData['residue name']
del allData['position']
del allData['chain']

#turn data2 into a 2d numpy array
arr_2d = allData.to_numpy()

#taking the cosine function of every data point, used for assignment 2D
#for i in range(len(arr_2d[:,0]-1)):
#    arr_2d[i,0]=math.cos(math.radians(arr_2d[i,0]))
#    arr_2d[i,1]=math.cos(math.radians(arr_2d[i,1]))

# For loop for trying out different values of K clusters.
for k in range(2,10):
  clusters = KMeans(n_clusters=k, random_state=0)
  clusterLabeler = clusters.fit_predict(arr_2d)
  # Calculating the sillhoutte-score to find the average of each sample to find the density
  avgSillhouette = silhouette_score(arr_2d, clusterLabeler)
  print(
        "For k number of clusters =",
        k,
        "Average silhouette score is:",
        avgSillhouette,
    )

