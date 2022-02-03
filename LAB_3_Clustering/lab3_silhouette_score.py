import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data200['residue name']
del data200['position']
del data200['chain']

#turn data200 into a 2d numpy array
arr_2d = data200.to_numpy()

 
for k in range(2,10):
  clusterer = KMeans(n_clusters=k, random_state=0)
  cluster_labels = clusterer.fit_predict(arr_2d)

  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(arr_2d, cluster_labels)
  print(
        "For k-clusters =",
        k,
        "The average silhouette score is then :",
        silhouette_avg,
    )

