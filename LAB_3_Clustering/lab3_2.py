import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)


#delete columns we dont want so we can create 2d array later
del data200['residue name']
del data200['position']
del data200['chain']

# Convert to simple lists that is used to plot the clusters in the first assignment.
phi = data200['phi'].tolist()
psi = data200['psi'].tolist()

#turn data200 into a 2d numpy array
arr_2d = data200.to_numpy()