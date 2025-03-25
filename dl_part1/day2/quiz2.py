import pandas as pd
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



cancer = load_breast_cancer()
# print(cancer.keys())
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target
# df.info()

cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean concavity', 'worst radius', 'worst perimeter', 'worst concavity',
        'worst texture']


class Logistic_Cancer(nn.Module):
    def __init__(self):
        super(Logistic_Cancer, self).__init__()
        self.linear = nn.Linear(len(cols),30,bias=True)
        self.linear2 = nn.Linear(30,20)
        self.linear3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        y = self.linear(x)
        y = self.linear2(y)
        y = self.linear3(y)
        y = self.sigmoid(y)

        return y

def prepare_datasets():
    x = torch.tensor(df[cols].values,dtype=torch.float32)
    y = torch.tensor(df[['class']].values,dtype=torch.float32)
    x_tr,x_t,y_tr,y_t = train_test_split(x,y,train_size=0.8,random_state=42)
    dataset = TensorDataset(x_tr,y_tr)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

    return dataloader,x_t,y_t

model = Logistic_Cancer()
optimizer = optim.Adam(model.parameters(),lr=0.00005)
loss_func = nn.BCELoss()

dataloader,x_test,y_test = prepare_datasets()
# print(x_train.shape,x_test.shape)

for epoch in range(5001):
    for step,batch in enumerate(dataloader):
        x_train, y_train = batch
        hypothesis = model(x_train)
        loss = loss_func(hypothesis,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 and step % 10 == 0:
            prediction = model(x_test) > torch.FloatTensor([0.5])
            correction_prediction = prediction.float() == y_test
            accuracy = correction_prediction.sum().item() / len(correction_prediction)
            print(f'epoch {epoch}, step:{step}, loss:{loss.item():.4f}, accuracy:{accuracy*100:2.4f}')