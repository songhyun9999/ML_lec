from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# x,y = make_moons(n_samples=500, noise=0.3, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=42)

from sklearn.ensemble import AdaBoostClassifier

###### Boosting #######
### 순차적으로 학습을 진행하며 처음 모델의 예측으로 부터
### learning rate만큼 다음 모델의 가중치를 수정
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                              n_estimators=300,
#                              random_state=42,
#                              learning_rate=0.3)
#
# ada_clf.fit(x_train,y_train)
# y_pred = ada_clf.predict(x_test)
# print(accuracy_score(y_test,y_pred))


from sklearn.tree import DecisionTreeRegressor
import numpy as np

x = np.random.rand(100,1) - 0.5
y = 3 * x[:,0]**2 + 0.05 + np.random.randn(100)
# print(x.shape,y.shape)

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(x,y)
y2 = y - tree_reg1.predict(x)

tree_reg2 = DecisionTreeRegressor(max_depth=2,random_state=42)
tree_reg2.fit(x,y2)
y3 = y2 - tree_reg2.predict(x)

tree_reg3 = DecisionTreeRegressor(max_depth=2,random_state=42)
tree_reg3.fit(x,y2)

x_new = np.array([[0.8]])

y_pred = sum(tree.predict(x_new) for tree in (tree_reg1,tree_reg2,tree_reg3))
print(y_pred)


from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=300,learning_rate=0.8)
gbrt.fit(x,y)
print(gbrt.predict(x))