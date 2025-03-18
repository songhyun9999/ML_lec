import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fish = pd.read_csv('https://bit.ly/fish_csv')
fish.info()
print(fish.head(10))
print(fish.Species.unique())

fish_input = fish.iloc[:,1:].values
print(fish_input)

fish_target = np.array(fish['Species'])
print(fish_target)

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_input,fish_target,random_state=42)

from sklearn.preprocessing import StandardScaler

ss= StandardScaler()
ss.fit(fish_input)
train_scaled = ss.transform(train_input)
# print(train_scaled[1])
test_scaled = ss.transform(test_input)
print(test_scaled[1])

# from sklearn.neighbors import KNeighborsClassifier
#
# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(train_scaled,train_target)
# print(f'knn model (train data)accuracy : {kn.score(train_scaled,train_target)}')
# print(f'knn model (test data)accuracy : {kn.score(test_scaled,test_target)}')
# print(kn.classes_)
#
# proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba,4))

bream_smelt_indexes = (train_target=='Bream') | (train_target=='Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
print(train_bream_smelt)
print(target_bream_smelt)
print()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print()
print(lr.predict_proba(train_bream_smelt[:5]))
print()

lr = LogisticRegression(C=30, max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
