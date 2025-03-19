from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x,y = make_moons(n_samples=5000, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(random_state=42)
svm_clf = SVC(probability=True,random_state=42) # 확률로 출력하여 soft 방식에 사용

##### 여러 모델에 대한 투표를 진행하여 분류
##### hard : 모든 모델에 대한 분류를 진행하여 다수결 투표 진행
##### soft : 모든 모델에서 각 레이블에 대한 확률 값을 구하고 각각 합하여 가장 큰 값을 사용
voting_clf = VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    # voting='hard'
    voting='soft'
)

from sklearn.metrics import accuracy_score

for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))




