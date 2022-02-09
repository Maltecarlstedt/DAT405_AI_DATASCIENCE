import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("data_all.csv", header = 0)


PROdata = df[df["residueName"] == "PRO"]
GLYdata = df[df["residueName"] == "GLY"]

PROX = PROdata[["phi", "psi"]].to_numpy()
GLYX = GLYdata[["phi", "psi"]].to_numpy()


plt.title("Phi and psi combinations of GLY")
plt.ylabel("psi")
plt.xlabel("phi")
plt.scatter(GLYX[:,0],GLYX[:,1])
#plt.scatter(PROX[:,0],PROX[:,1])

plt.show()

