__author__ = "Malte Carlstedt, Johan Östling"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Read from csv-files
df = pd.read_csv("data_all.csv", header = 0)

# Retrieve data from columns which match PRO and GLY
PROdata = df[df["residueName"] == "PRO"]
GLYdata = df[df["residueName"] == "GLY"]

# Retrieving Phi and Psi values
GLYX = PROdata[["phi", "psi"]].to_numpy()
GLYX = GLYdata[["phi", "psi"]].to_numpy()

#adding titles to the plot
plt.title("Phi and psi combinations of GLY")
plt.ylabel("psi")
plt.xlabel("phi")
#plt.scatter(GLYX[:,0],GLYX[:,1])
#plt.scatter(PROX[:,0],PROX[:,1])


# Using DBSCAN to compute our data
db = DBSCAN(eps=25, metric="euclidean", min_samples=55).fit(GLYX)

# Find number of clusters by checking their given labels and allowing no duplicates => gives clusters.
numberOfClusters =len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
print('Estimated no. of clusters: ', numberOfClusters)
# Find outliers through which label they were given.
numberOfNoisePoints = list(db.labels_).count(-1)
print('Estimated number of of noise points: ', numberOfNoisePoints)

# Scatter our 2d array by column 0 and 1. Colouring according to their labels.
plt.scatter(GLYX[:, 0], 
            GLYX[:, 1], 
            c=db.labels_)


# Scatter over our plots that are considered as outliers by checking their label. Probably not the best complexity but it works. Colouring all outliers black.
for i in range(len(db.labels_)-1):
    if db.labels_[i]==-1:
        plt.scatter(GLYX[i, 0], 
                    GLYX[i, 1], 
                    c='black')



plt.title("Min_samples = 55 and Epsilon = 25 for GLY")
plt.show()
