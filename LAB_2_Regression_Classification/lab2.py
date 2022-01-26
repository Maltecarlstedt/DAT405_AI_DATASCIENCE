import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

houses = pd.read_csv("houses.csv", header = 0)

#area = houses['area'].tolist()
#price = houses['price'].tolist()

# Chooses Column 0 and reshape it to fit into an numpy array
x = houses.iloc[:, 0].values.reshape(-1, 1) 
# Chooses column 1 and reshapre it to fit into an numpy array
y = houses.iloc[:, 1].values.reshape(-1, 1) 

# Calculate linear regression, which is trying to fit a straight line in our dataset. Adjusting to our data.
model = LinearRegression().fit(x, y)

# Plots our x and y values. That is our area and price values.
#plt.scatter(x, y)
# Prints the K-value (slope)
#print(model.coef_)
# Prints the m-value (intercept value)
#print(model.intercept_)
#plt.ylabel("Price")
#plt.xlabel("Area")
#plt.show();

# Choosing to plot for x values between 0 to 300. With 1000 evenly spaced values.
xfit = np.linspace(0, 300, 1000) 
# Gives the y value of f(xFit)
yfit = model.predict(xfit[:, np.newaxis])
# Givest the predicted cost for houses of size 100, 150, 200 square meters. 
#print(model.predict([[100]])) 
#print(model.predict([[150]]))
#print(model.predict([[200]]))

print(model.predict(x))

plt.ylabel('Residuals')
plt.xlabel('area [m^2]')
plt.scatter(x,y-model.predict(x))
plt.axhline(y=0)
plt.show()

#print(yfit.tolist())
#residuals = getResiduals(y.tolist() , yfit.tolist())
#print(residuals)
plt.scatter(x, y)
plt.plot(xfit, yfit)

#plt.show()