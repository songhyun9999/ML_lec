import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


df = pd.read_csv('diabetes.csv',header=None)
# print(df.head(5))

class Diabetes_Dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])

        return x,y


class Diabetes_Classifier(nn.Module):
    def __init__(self):
        super(Diabetes_Classifier,self).__init__()
        self.input_layer = nn.Linear(8,40,bias=True)
        self.hidden_layer = nn.Sequential(
            nn.Linear(40,10),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self.hidden_layer(self.input_layer(x))

        return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1].values,df.iloc[:,-1:].values,random_state=42)
    x_test, y_test = torch.FloatTensor(x_test).to(device), torch.FloatTensor(y_test).to(device)
    dataset = Diabetes_Dataset(x_train,y_train)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

    model = Diabetes_Classifier().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_func = nn.BCELoss()

    for epoch in range(5001):
        losses = 0
        n=1

        for step,batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x_train, y_train = batch
            hypothesis = model(x_train)
            loss = loss_func(hypothesis,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                losses += loss.item()
                n = n+1

        if epoch%500 == 0:
            losses = losses/n
            pred = (model(x_test) > torch.FloatTensor([0.5]).to(device)).float()
            accuracy = (pred == y_test).sum().item() / len(pred)
            print(f'epoch {epoch}  loss:{losses:.4f}, accuracy:{accuracy*100:2.2f}')


