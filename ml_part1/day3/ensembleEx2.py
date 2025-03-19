from sklearn.datasets import load_digits # 손글씨 데이터
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(5)

mnist = load_digits()
print(mnist.keys())
# print(mnist.DESCR)
print(mnist.data.shape)
print(mnist.target)
# print(mnist.images.shape)

x_train, x_test, y_train, y_test = train_test_split(mnist.data,mnist.target,
                                                    train_size=0.8,random_state=42)

dtree = DecisionTreeClassifier(max_depth=8,random_state=35)
dtree.fit(x_train,y_train)
dtree_predicted = dtree.predict(x_test)

knn = KNeighborsClassifier(n_neighbors=299)
knn.fit(x_train,y_train)
knn_predicted = knn.predict(x_test)

svm = SVC(C=0.1,gamma=0.003,probability=True,random_state=35)
svm.fit(x_train,y_train)
svm_predicted = svm.predict(x_test)

print('-----accuracy-----')
print(f'dtree : {accuracy_score(y_test,dtree_predicted)}')
print(f'knn : {accuracy_score(y_test,knn_predicted)}')
print(f'svm : {accuracy_score(y_test,svm_predicted)}')

voting_clf = VotingClassifier(
    estimators=[('dt',dtree),('knn',knn),('svm',svm)],
    voting='hard'
)

hard_voting_predicted = voting_clf.fit(x_train,y_train).predict(x_test)
print(f'voting(hard) : {accuracy_score(y_test,hard_voting_predicted)}')

voting_clf2 = VotingClassifier(
    estimators=[('dt',dtree),('knn',knn),('svm',svm)],
    voting='soft'
)

soft_voting_predicted = voting_clf2.fit(x_train,y_train).predict(x_test)
print(f'voting(soft) : {accuracy_score(y_test,soft_voting_predicted)}')