import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv("WineData.csv")

points = df_wine["points"]
points_lite = points.head(200)

price = df_wine["price"]
price_lite = price.head(200)

# points_train, points_test, price_train, price_test = train_test_split(points_lite.values.reshape(-1,1), price_lite.values.reshape(-1,1), test_size=0.5, random_state=35)
points_train, points_test, price_train, price_test = train_test_split(points.values.reshape(-1,1), price.values.reshape(-1,1), test_size=0.5, random_state=35)

regr_train = linear_model.LinearRegression().fit(points_train, price_train)
price_pred = regr_train.fit(points_train, price_train).predict(points_test)

plt.scatter(points_train, price_train, color='green')
plt.scatter(points_train, regr_train.predict(points_train), color='red')
plt.scatter(points_test[:10,:], price_test[:10], color='black')
plt.title('Linear regression for Wine Price')
plt.xlabel('points')
plt.ylabel('price')
plt.show()

regr = linear_model.LinearRegression()
# regr.fit(points_lite.values.reshape(-1,1),price_lite)
regr.fit(points.values.reshape(-1,1),price)
# plt.plot(points_lite, regr.predict(points_lite.values.reshape(-1,1)), color = "blue")
plt.plot(points, regr.predict(points.values.reshape(-1,1)), color = "blue")
plt.xlabel("Points")
plt.ylabel("Price")
# plt.scatter(points_lite, price_lite, color = "red")
plt.scatter(points, price, color = "red")
plt.show()

need_prediction = [89.3,88.6,90.1]
for elem in need_prediction:
    print(regr.predict([[elem]]))
