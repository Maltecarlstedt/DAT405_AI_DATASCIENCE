__author__ = "Malte Carlstedt, Johan Östling"
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv-file
data = pd.read_csv("data_all.csv", header = 0)

# turn data into a 2d numpy array
X = data[["phi", "psi"]].to_numpy()

# Computing and fitting using DBSCAN.
db = DBSCAN(eps=12, metric="euclidean", min_samples=70).fit(X)

# Retrieving all noisepoints in a list (removing the datapoints that are not considered outliers)
noisePoints = data[db.labels_==-1]

# Styling for easier look
plt.style.use('ggplot')
plt.title("Bargraph for how many noise point relates to each residue")
# Plotting each aminoacids depending on how many time a certain residue is in the noisePoints list. Plotting as as a barchart with colouring for easier look.
aminoAcids = noisePoints["residueName"].value_counts().plot(kind = "bar", color = ["green","red"])

plt.show()