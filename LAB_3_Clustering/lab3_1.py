import pandas as pd
import matplotlib.pyplot as plt


#save all data in file data_200.csv except the first row
data200 = pd.read_csv("data_200.csv", header = 0)

#saves columns that we want to scatter plot in lists
phi = data200['phi'].tolist()
psi = data200['psi'].tolist()


#plotting and adding titles
plt.title("Phi and psi combinations")
plt.ylabel("psi")
plt.xlabel("phi")
plt.scatter(phi,psi)
plt.show()