import matplotlib.pyplot as plt
import numpy as np

# Data from 
denoisedScores = [0.9649999737739563, 0.9431999921798706, 0.9072999954223633,0.7696999907493591, 0.37630000710487366, 0.09749999642372131]
seasonedScores = [0.9639999866485596, 0.9182999730110168, 0.849399983882904, 0.6966000199317932, 0.366100013256073, 0.1062999963760376]

noise = [0, 0.2, 0.4, 0.6, 0.8, 1]

print(np.linspace(0,1, 20))
plt.scatter(noise, denoisedScores)
plt.scatter(noise, seasonedScores)
plt.show()