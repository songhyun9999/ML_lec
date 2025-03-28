import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

sdata = np.array([
    [166,59],
    [176,75.7],
    [171, 62.1],
    [173, 70.4],
    [169, 60.1],
])

x = sdata[:,0]
y = sdata[:,1]

x = x - x.mean()
y = y - y.mean()
# plt.scatter(x,y,s=50)
# plt.xlabel('신장(cm)')
# plt.ylabel('체중(kg)')
# plt.show()

import torch

x_data = torch.tensor(x,dtype=torch.float32)
y_data = torch.tensor(y,dtype=torch.float32)

w = torch.tensor(1,requires_grad=True,dtype=torch.float32)
b = torch.tensor(1,requires_grad=True,dtype=torch.float32)

def predict(x):
    return w * x + b

def mse(h,y):
    return ((h-y)**2).mean()

history = np.zeros((0,2))

for epoch in range(500):
    hypothesis = predict(x_data)
    # print(hypothesis)

    loss = mse(hypothesis,y_data)
    # print(loss)

    loss.backward()
    # print(w.grad)
    # print(b.grad)
    lr = 0.00001
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
    if epoch%10 == 0:
        item = np.array([epoch,loss.item()])
        history = np.vstack((history,item))
        print(f'epoch {epoch:<2}\tloss:{loss.item():.4f}')


output = predict(x_data)
print(f'final output : {output.detach().numpy()}')

x_plot = torch.linspace(-7.5,7.5,50)
y_plot = predict(x_plot)

with torch.no_grad():
    plt.plot(x_plot,y_plot,'r--')
    plt.scatter(x,y,s=50)
    plt.xlabel('신장(cm)')
    plt.ylabel('체중(kg)')
    plt.show()

# print(w)
# print(b)
# print(w.grad)
# print(b.grad)

# plt.plot(history[:, 0], history[:, 1], 'b')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 그래프(손실)')
# plt.show()
