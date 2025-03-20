import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
print(cancer.target)

x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,random_state=10)

#### Stacking 예제 ####

model1 = DecisionTreeClassifier()
model1.fit(x_train,y_train)
val_pred1 = model1.predict(x_val)
test_pred1 = model1.predict(x_test)

val_pred1 = pd.DataFrame(val_pred1)
test_pred1 = pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2 = model2.predict(x_val)
test_pred2 = model2.predict(x_test)

val_pred2 = pd.DataFrame(val_pred2)
test_pred2 = pd.DataFrame(test_pred2)

x_val = pd.DataFrame(x_val)
x_test = pd.DataFrame(x_test)

df_val = pd.concat([x_val,val_pred1,val_pred2],axis=1)
df_test = pd.concat([x_test,test_pred1,test_pred2],axis=1)

model = LogisticRegression(max_iter=1500)
model.fit(df_val,y_val)
print('accuracy: ',model.score(df_test,y_test))

