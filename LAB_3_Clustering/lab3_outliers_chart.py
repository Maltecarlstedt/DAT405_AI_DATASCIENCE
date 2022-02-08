import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data_all.csv", header = 0)

#delete columns we dont want so we can create 2d array later


#turn data200 into a 2d numpy array
X = data[["phi", "psi"]].to_numpy()


db = DBSCAN(eps=12, metric="euclidean", min_samples=70).fit(X)

noisePoints = data[db.labels_==-1]

# Styling for easier look
plt.style.use('ggplot')
plt.title("Bargraph for how many noise point relates to each residue")
aminoAcids = noisePoints["residueName"].value_counts().plot(kind = "bar", color = ["green","red"])


plt.show()