import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn

l1 = nn.Linear(784, 128)
l2 = nn.Linear(128, 10)
relu = nn.ReLU()

inputs = torch.randn(100, 784)

m1 = l1(inputs)
m2 = relu(m1)
outputs = l2(m2)

# print(m1)
# print()
# print(m2)
# print(outputs)

print(inputs.shape)
print(outputs.shape)

net = nn.Sequential(
    l1,
    relu,
    l2
)
output2 = net(inputs)
print(output2)

np.random.seed(126)
x = np.random.randn(100,1)
y = x ** 2 + np.random.randn(100,1) + 0.1

x_train = x[:50,:]
x_test = x[50:,:]
y_train = y[:50,:]
y_test = y[50:,:]

# plt.scatter(x_train, y_train, c='b', label='훈련 데이터')
# plt.scatter(x_test, y_test, c='k', marker='x', label='검증 데이터')
# plt.legend(loc='best')
# plt.show()

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        y = self.fc1(x)
        return y

model = Net()
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
history = np.zeros((0,2))

for epoch in range(10000):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'epoch:{epoch+1}, loss:{loss.item():.4f}')

# with torch.no_grad():
#     prediction = model(x_test)
#     plt.title('은닉층 없음, 활성화 함수 없음')
#     plt.scatter(x_test[:, 0], prediction[:, 0], c='b', label='예측값')
#     plt.scatter(x_test[:, 0], y_test[:, 0], c='k', marker='x', label='정답')
#     plt.legend(loc='best')
#     plt.show()


# prediction = model(x_test)
# plt.title('은닉층 없음, 활성화 함수 없음')
# plt.scatter(x_test[:, 0].detach(), prediction[:, 0].detach(), c='b', label='예측값')
# plt.scatter(x_test[:, 0].detach(), y_test[:, 0].detach(), c='k', marker='x', label='정답')
# plt.legend(loc='best')
# plt.show()

prediction = model(x_test)
plt.title('은닉층 없음, 활성화 함수 없음')
plt.scatter(x_test[:, 0].data, prediction[:, 0].data, c='b', label='예측값')
plt.scatter(x_test[:, 0].data, y_test[:, 0].data, c='k', marker='x', label='정답')
plt.legend(loc='best')
plt.show()

