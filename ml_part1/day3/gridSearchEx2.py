from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

x,y = make_moons(random_state=42,noise=0.15,n_samples=10000)

train_x, test_x, train_y, test_y = train_test_split(x,y,random_state=42,train_size=0.8)

params = {'max_leaf_nodes':range(2,100,1),
          'min_samples_split':range(2,11,1)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)

gs.fit(train_x,train_y)
print(gs.best_params_)
print(gs.score(test_x,test_y))

best_estim = gs.best_estimator_
ypred = best_estim.predict(test_x)

from sklearn.metrics import accuracy_score

print(f'accuracy : {accuracy_score(test_y,ypred)}')

