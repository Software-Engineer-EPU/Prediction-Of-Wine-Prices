import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df_house = pd.read_csv("WineData.csv")

points = df_house["points"]
# points_full = df_house["points"]
# points = points_full.head(100)
# print (points.head(10))

price = df_house["price"]
# price_full = df_house["price"]
# price = price_full.head(100)
# print (price.head(10))


points = PCA(1).fit_transform(points.values.reshape(-1,1))
print (points[:10])

points_train, points_test, price_train, price_test = train_test_split(points, price, test_size=0.5, random_state=35)
regr = linear_model.LinearRegression().fit(points_train, price_train)
price_pred = regr.predict(points_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Bias: \n', regr.intercept_)
# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(price_test, price_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(price_test, price_pred))

plt.scatter(points_train, price_train, color='green')
plt.scatter(points_train, regr.predict(points_train), color='red')
plt.scatter(points_test[:10,:], price_test[:10], color='black')
plt.title('Linear regression for Wine Price')
plt.xlabel('points')
plt.ylabel('price')
plt.show()

plt.plot([min(price_test), max(price_test)],[min(price_pred),max(price_pred)])
plt.scatter(price_test, price_pred, color='red')
plt.title('Compare')
plt.xlabel('price_test')
plt.ylabel('price_pred')
plt.show()