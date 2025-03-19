from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
print(cancer.keys())
# print(cancer.DESCR)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=42)

rf_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1,random_state=42)
rf_clf.fit(x_train,y_train)
y_pred = rf_clf.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy ', accuracy_score(y_test,y_pred))