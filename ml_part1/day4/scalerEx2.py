from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


x,_= make_blobs(n_samples=50,centers=5,random_state=4,cluster_std=2)
x_train, x_test = train_test_split(x,random_state=5,test_size=0.1)

##### scale 방법 #####
### scale은 traindata 기준으로 진행한다
### scaler를 traindata에 맞춰 바꾸고 test data 역시 traindata 기준으로 변형

fig,axes = plt.subplots(1,3,figsize=(10,3))
axes[0].scatter(x_train[:,0],x_train[:,1], c='orange',label='train data',s=60)
axes[0].scatter(x_test[:,0],x_test[:,1], c='blue',label='test data',s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('real data')


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

axes[1].scatter(x_train_scaled[:,0],x_train_scaled[:,1], c='orange',label='x_train_scaled data',s=60)
axes[1].scatter(x_test_scaled[:,0],x_test_scaled[:,1], c='blue',label='x_test_scaled data',s=60)
axes[1].legend(loc='upper left')
axes[1].set_title('scaled data')


test_scaler = MinMaxScaler()
test_scaler.fit(x_test)
x_test_scaled_badly = test_scaler.transform(x_test)

axes[2].scatter(x_train_scaled[:,0],x_train_scaled[:,1], c='orange',label='x_train_scaled data',s=60)
axes[2].scatter(x_test_scaled_badly[:,0],x_test_scaled_badly[:,1], c='blue',label='x_test_scaled_badly data',s=60)
axes[2].legend(loc='upper left')
axes[2].set_title('misadjusted data')

plt.show()