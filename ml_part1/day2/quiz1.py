import pandas as pd
import numpy as np


## prepare datasets
wine = pd.read_csv('https://bit.ly/wine-date')
# wine.info()
# print(wine.head(10))
# print(wine['class'].unique())

x = np.array(wine.iloc[:,:3])
y = np.array(wine.iloc[:,-1])
# print(x,y)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=0.8,random_state=42)

ss = StandardScaler()
ss.fit(train_x)

train_x = ss.transform(train_x)
test_x = ss.transform(test_x)

# print(train_x[:10])

## logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(train_x,train_y)

# print(lr.predict(test_x[:7]))
# print()
# print(lr.predict_proba(test_x[:7]))
# print()
# print(test_y[:7])
# print()
print(f'logistic regression score(train) : {lr.score(train_x,train_y)}')
print(f'logistic regression score(test) : {lr.score(test_x,test_y)}')
print()

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42,max_depth=4)
dt.fit(train_x,train_y)

# print(dt.predict(test_x[:7]))
# print(test_y[:7])
print(f'decision tree score(train) : {dt.score(train_x,train_y)}')
print(f'decision tree score(test) : {dt.score(test_x,test_y)}')

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

