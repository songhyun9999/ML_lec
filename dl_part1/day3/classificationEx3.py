import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split


wine = load_wine()
# print(wine)

df = pd.DataFrame(wine.data,columns=wine.feature_names)
# df.info()

wine_data = wine.data[0:130]
wine_target = wine.target[0:130]
# print(wine_target)
wine_data, wine_target = torch.FloatTensor(wine_data), torch.LongTensor(wine_target)
# wine_target = wine_target.unsqueeze(1)
# print(wine_target.shape)
x_train, x_test, y_train, y_test = train_test_split(wine_data,wine_target,random_state=48,test_size=0.2)
# print(x_train.shape)
dataset = TensorDataset(x_train,y_train)
dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel,self).__init__()
        self.fc1 = nn.Linear(13,96)
        self.fc2 = nn.Linear(96,96)
        self.fc3 = nn.Linear(96,64)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,2)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        y = self.fc6(out)

        return y


model = ClassificationModel()
# optimizer = optim.SGD(model.parameters(),lr=0.01)
optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1001):
    total_loss = 0

    for step,batch in enumerate(dataloader):
        optimizer.zero_grad()
        x_train, y_train = batch
        hypothesis = model(x_train)

        loss = loss_func(hypothesis,y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch%100 == 0:
        print(f'epoch {epoch}, total_loss:{total_loss:.4f}')


print()
model.eval()
with torch.no_grad():
    pred = torch.max(model(x_test),dim=1)[1]
    accuracy = (pred==y_test).float().mean()
    print(f'accuracy :{accuracy.item():.4f}')