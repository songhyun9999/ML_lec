import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def prepare_datasets():
    wine = pd.read_csv('https://bit.ly/wine-date')
    data = torch.tensor(wine[['alcohol','sugar','pH']].values,dtype=torch.float32)
    target = torch.tensor(wine[['class']].values,dtype=torch.float32)


    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=42)
    dataset = TensorDataset(x_train,y_train)
    dataloader = DataLoader(dataset,batch_size=256,shuffle=True)

    return dataloader, x_test, y_test


class BinaryClassification_LogisticModel(nn.Module):
    def __init__(self):
        super(BinaryClassification_LogisticModel,self).__init__()
        self.inputlayer = nn.Linear(3,20,bias=True)
        self.hidden_layer = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    def forward(self,x_data):
        y = self.hidden_layer(self.inputlayer(x_data))

        return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader,x_test,y_test = prepare_datasets()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = BinaryClassification_LogisticModel().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_func = nn.BCELoss()
    for epoch in range(1001):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x_train, y_train = batch

            hypothesis = model(x_train)
            loss = loss_func(hypothesis,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 ==0:
            pred = (model(x_test) > 0.5).float()
            accuracy = (pred == y_test).sum().item()/len(y_test)
            print(f'epoch {epoch}, total_loss:{total_loss:.4f},accuracy:{accuracy*100:2.2f}')


