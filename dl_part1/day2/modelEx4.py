import torch
import torch.nn as nn
import torch.optim as optim


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.tensor(x_data,dtype=torch.float32)
y_train = torch.tensor(y_data,dtype=torch.float32)

class LogisticClass(nn.Module):
    def __init__(self):
        super(LogisticClass,self).__init__()
        self.linear = nn.Linear(2,1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.linear(x)
        y = self.sigmoid(y)

        return y

import torch.nn.functional as F

# loss_func = nn.BCELoss()
model = LogisticClass()
optimizer = optim.SGD(model.parameters(),lr=1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = F.binary_cross_entropy(hypothesis,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis > torch.FloatTensor([0.5])
        correction_prediction = prediction.float() == y_train
        accuracy = correction_prediction.sum().item() / len(correction_prediction)
        print(f'epoch{epoch+1} loss:{loss.item():.4f} accuracy:{accuracy:2.2f}')

# print()
# hypothesis = model(x_train)
# print(hypothesis)
# prediction = hypothesis > torch.FloatTensor([0.5])
# print(prediction)
