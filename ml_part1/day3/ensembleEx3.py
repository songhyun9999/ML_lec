from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


x,y = make_moons(n_samples=500, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=42)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

##### Bagging & pasting
### bagging : 데이터의 중복 허용 (복원 추출)
### pasting : 데이터의 중복 허용 X (비복원 추출)

# bagging
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, # dtree 500개 생성
    max_samples=100, # 각 모델에 사용할 데이터 sample의 개수를 100개로 제한
    bootstrap=True, # bagging 사용 여부
    random_state=42
)

bag_clf.fit(x_train,y_train)
y_pred = bag_clf.predict(x_test)
print('bagging accuracy : ',accuracy_score(y_test,y_pred))

# pasting
bag_clf2 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, # dtree 500개 생성
    max_samples=100, # 각 모델에 사용할 데이터 sample의 개수를 100개로 제한
    bootstrap=False, # bagging 사용 여부
    random_state=42
)

bag_clf2.fit(x_train,y_train)
y_pred = bag_clf2.predict(x_test)
print('pasting accuracy : ',accuracy_score(y_test,y_pred))


