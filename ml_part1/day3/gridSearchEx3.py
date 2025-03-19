import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_test.csv')
# train.info()
# print(train)

## Prepare Datasets
train_x, train_y = train[['3P','BLK']], train['Pos']
test_x, test_y = test[['3P','BLK']], test['Pos']


## Set GridSearchCV
params = {'kernel':['rbf'],
          'C':[0.01,0.1,1,10,100],
          'gamma':[0.0001,0.001,0.01,0.1,1]}
gs = GridSearchCV(SVC(random_state=42),params,n_jobs=-1)


# training & test
gs.fit(train_x,train_y.values.ravel())
ypred = gs.predict(test_x)


print(gs.best_params_)
print(gs.score(test_x,test_y.values.ravel()))
print('accuracy : ',accuracy_score(test_y.values.ravel(),ypred))

