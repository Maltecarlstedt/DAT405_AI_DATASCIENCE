__author__ = "Malte Carlstedt, Johan Östling"
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv file
data = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later
del data['residueName']
del data['position']
del data['chain']

#turn data into a 2d numpy array
X = data.to_numpy()

# 
db = DBSCAN(eps=14, metric="euclidean", min_samples=60).fit(X)

# Find number of clusters by checking 
numberOfClusters =len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
print('Estimated no. of clusters: ', numberOfClusters)
# Find outliers through which label they were given.
numberOfNoisePoints = list(db.labels_).count(-1)
print('Estimated number of of noise points: ', numberOfNoisePoints)

print("_______________________________________________")


plt.scatter(X[:, 0], 
            X[:, 1], 
            c=db.labels_)



# Recolor noise
for i in range(len(db.labels_)-1):
    if db.labels_[i]==-1:
        plt.scatter(X[i, 0], 
                    X[i, 1], 
                    c='black')



#plt.title("Min_samples = 70 and Epsilon = 12")
plt.show()
