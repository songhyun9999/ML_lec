import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.rc('font',family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(123)

from sklearn.datasets import load_iris

iris = load_iris()

x_org, y_org = iris.data, iris.target

print(x_org.shape, y_org.shape)

x_data = iris.data[:100,:2]
y_data = iris.target[:100]
# print(y_data)

# x_t0 = x_data[y_data==0]
# x_t1 = x_data[y_data==1]
# plt.scatter(x_t0[:,0],x_t0[:,1],marker='x',c='b',label='0 (setosa)')
# plt.scatter(x_t1[:,0],x_t1[:,1],marker='o',c='k',label='1 (versicolor)')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')
# plt.legend(loc='best')
# plt.show()


from sklearn.model_selection import train_test_split

# 텐서 변환
x_data, y_data = torch.tensor(x_data).float(),torch.tensor(y_data).float()
y_data = y_data.unsqueeze(1)
# train, test 분리
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3,shuffle=True,random_state=42)

import torch.nn.functional as F
class LogisticClassificationModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(LogisticClassificationModel,self).__init__()
        self.fc1 = nn.Linear(input_size,output_size)

        self.fc1.weight.data.fill_(1.0)
        self.fc1.bias.data.fill_(1.0)

    def forward(self,x):
        y = F.sigmoid(self.fc1(x))

        return y



model = LogisticClassificationModel(x_train.shape[1],1)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-2)
history = np.zeros((0,5))
for epoch in range(10001):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        with torch.no_grad():
            pred = hypothesis > torch.tensor([0.5])
            accuracy = (pred==y_train).sum().item()/len(pred)

            val_out = model(x_test)
            val_loss = loss_func(val_out,y_test)
            pred_val = val_out> torch.tensor([0.5])
            accuracy_val = (pred_val == y_test).sum().item() / len(pred_val)
            print(f'epoch {epoch}\tloss:{loss.item():.5f}\taccuray:{accuracy*100:2.2f}'
                  f'\tvalidation_loss:{val_loss.item():.5f}\taccuracy:{accuracy_val*100:2.2f}')
            history = np.vstack((history, np.array([epoch,loss.item(),accuracy,val_loss,accuracy_val])))

print()
plt.plot(history[:,0],history[:,1],'b',label='훈련데이터')
plt.plot(history[:,0],history[:,3],'b',label='훈련데이터')
plt.xlabel('반복 횟수')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

