
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn

torch.manual_seed(123)

# l1 = nn.Linear(1,1)
# print(l1)
# print()
#
# for param in l1.named_parameters():
#     print(param[0])
#     print(param[1])
# print()
#
# l2 = nn.Linear(2, 1)
# print(l2.weight)
# nn.init.constant_(l2.weight, 1.0)
# nn.init.constant_(l2.bias, 2.0)
# print()
# print(l2.weight)
# print(l2.bias)

data_url = 'http://lib.stat.cmu.edu/datasets/boston'

df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                          'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
'''
 'CRIM':범죄율(1인당 범죄수)
 'ZN': 25,000평방피트 이상 주거지역 비율(%)
 'INDUS':비소매업 종사자 비율
 'CHAS':찰스강 인접 여부(1:인접, 0:비인접)
 'NOX' :일산화탄소 농도
 'RM' : 주택당 평균 방의 개수
 'AGE' :1940년 이전에 지어진 주택 배율
 'DIS' :보스턴 주요 고용센터까지의 거리
 'RAD' :고속도로 접근 용이성 지수
 'TAX' :재산세 비율
 'PTRATIO':학생 교사 비율
 'B': 흑인 거주 비율
 'LSTAT': 저소득층 비율
'''

x_org = np.hstack([df.values[::2, :],
                   df.values[1::2, :2]])
print(x_org.shape)
yt = df.values[1::2, 2] #보스톤 집 값
print(yt.shape)

x = x_org[:, feature_names=='RM']
print(x)

# plt.scatter(x, yt, s=10, c='b')
# plt.xlabel('방 개수')
# plt.ylabel('집 가격')
# plt.title('방 개수와 집 가격의 산포도')
# plt.show()

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net,self).__init__()
        self.l1 = nn.Linear(input_size,output_size)
        # l1.weight.data.fill_(1.0)
        # l1.bias.data.fill_(1.0)
        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    def forward(self,x):
        return self.l1(x)

model = Net(x.shape[1], 1)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.tensor(x).float()
y_data = torch.tensor(yt).float()
y_data = y_data.view((-1,1))
print(y_data.shape)

for epoch in range(50000):
    optimizer.zero_grad()
    hypothesis = model(x_data)
    loss = loss_func(hypothesis, y_data) / 2
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch:{epoch+1} loss:{loss.item():.4f}')

x_test = np.array((x.min(), x.max())).reshape(-1,1)
x_test = torch.tensor(x_test).float()

with torch.no_grad():
    prediction = model(x_test)

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('방 개수')
plt.ylabel('집 가격')
plt.plot(x_test.data, prediction.data, c='r')
plt.title('방 개수와 집 가격의 산포도')
plt.show()


