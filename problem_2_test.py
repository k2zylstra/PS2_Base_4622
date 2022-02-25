from problem_1_test import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import data
import numpy as np

house_prices = data.HousePrices()
#Workspace 2.1
#TODO: Discretize y_train and y_test
#BEGIN 
for t in [house_prices.y_train, house_prices.y_test]:
    for i in range(len(t)):
        if t[i] < 125000:
            t[i] = 0
        elif t[i] >= 125000 and t[i] < 160000:
            t[i] = 1
        elif t[i] >= 160000 and t[i] < 200000:
            t[i] = 2
        elif t[i] >= 200000:
            t[i] = 3 
#END
print(np.unique(house_prices.y_train), house_prices.X_train.shape)
print(np.unique(house_prices.y_test), house_prices.X_test.shape)

skdt = DecisionTreeClassifier(max_features=None)
dt = DecisionTree(max_depth=5, min_samples_split=2)
print("here")
skdt.fit(house_prices.X_train, house_prices.y_train)
dt.fit(house_prices.X_train, house_prices.y_train)
print("here2")
sk_score = skdt.score(house_prices.X_test, house_prices.y_test)
my_score = dt.score(house_prices.X_test, house_prices.y_test)
print("score", sk_score*100)
print("score", my_score*100)