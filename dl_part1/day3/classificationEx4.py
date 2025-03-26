import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

iris = pd.read_csv('iris.csv')
iris.info()
column_name = iris.Name.unique()

for idx,name in enumerate(column_name):
    iris[iris.Name==name] = idx

x_data = torch.tensor(iris.iloc[:,:-1].values,dtype=torch.float32)
y_data = torch.tensor(iris.iloc[:,-1],dtype=torch.long)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=42)

train_dataset = TensorDataset(x_train,y_train)
train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)

class IrisClassificationModel(nn.Module):
    def __init__(self):
        super(IrisClassificationModel,self).__init__()
        self.fc1 = nn.Linear(4,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16,8)
        self.fc5 = nn.Linear(8,3)

    def forward(self,x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = nn.functional.relu(self.fc4(out))
        y= self.fc5(out)
        return y


if __name__ == '__main__':
    model = IrisClassificationModel()
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1001):
        total_loss = 0

        for step,batch in enumerate(train_dataloader):
            x_train, y_train = batch

            hypothesis = model(x_train)
            loss = loss_func(hypothesis,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f'epoch {epoch}, total_loss:{total_loss:.4f}')

    print()
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
        # print(preds[:20])
        # print(y_test[:20])
        accuracy = (preds == y_test).sum().item()/len(preds)
        print(f"Accuracy: {accuracy:.2f}")

