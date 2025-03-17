import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

x,y = make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1.5,random_state=4)

# plt.figure(figsize=(7,7))
# plt.scatter(x[:,0],x[:,1],c=y,marker='*',edgecolors='k',s=80)
# plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=0)
print(x_train.shape)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn1 = KNeighborsClassifier(n_neighbors=1)

knn5.fit(x_train,y_train)
knn1.fit(x_train,y_train)

y_pred5 = knn5.predict(x_test)
y_pred1 = knn1.predict(x_test)

from sklearn.metrics import accuracy_score
print(f'accuracy with k=5 {accuracy_score(y_test,y_pred5)*100}')
print(f'accuracy with k=1 {accuracy_score(y_test,y_pred1)*100}')

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.scatter(x_test[:,0],x_test[:,1],c=y_pred5,marker='*',edgecolors='k',s=80)
plt.title('predicted values with k=5',fontsize=17)

plt.subplot(122)
plt.scatter(x_test[:,0],x_test[:,1],c=y_pred1,marker='*',edgecolors='k',s=80)
plt.title('predicted values with k=1',fontsize=17)
plt.show()


