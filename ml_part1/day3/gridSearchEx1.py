import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


wine = pd.read_csv('https://bit.ly/wine-date')

data = np.array(wine[['alcohol','sugar','pH']])
target = np.array(wine['class'])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data,target,train_size=0.8,random_state=42)


# params = {'min_impurity_decrease' : [0.0001,0.0002,0.0003,0.0004,0.0005]}
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
#              params,
#              n_jobs=-1)
#
# gs.fit(x_train,y_train)

# dt = gs.best_estimator_
# print(gs)
# print(dt.score(x_train,y_train))
# print(gs.best_params_)
# print(gs.cv_results_['mean_test_score'])

import numpy as np
params = {'min_impurity_decrease':np.arange(0.0001,0.001,0.0001),
          'max_depth':range(5,20,1),
          'min_samples_split':range(2,100,10)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(x_train,y_train)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
