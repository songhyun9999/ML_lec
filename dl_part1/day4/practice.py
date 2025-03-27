import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Subset
import numpy as np

# 랜덤값 고정
torch.manual_seed(777)

# Prepare MNist dataset
def prepare_datasets(batch_size):
    train_data = dset.MNIST(root='MNIST_data/',train=True,
                                download=True,transform=transforms.ToTensor())

    test_data = dset.MNIST(root='MNIST_data/', train=False,
                                download=True, transform=transforms.ToTensor())

    indices = list(range(len(train_data)))
    train_idx, valid_idx = train_test_split(indices,test_size=0.2,shuffle=True,random_state=42)

    train_data, valid_data = Subset(train_data,train_idx), Subset(train_data,valid_idx)

    train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(valid_data,batch_size=batch_size)
    test_data_loader = DataLoader(test_data,batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader

# Model 설계
class MNistClassificationModel_Linear(nn.Module):
    def __init__(self,input_size,output_size,dropout=0.2):
        super(MNistClassificationModel_Linear,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size,256,bias=True)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)

        self.dropout = 0.2

    def forward(self,x):
        x = x.view(-1, 784)
        out = F.relu(self.fc1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = F.relu(self.fc3(out))
        out = F.dropout(out,p=self.dropout,training=self.training)
        y = self.fc4(out)

        return y

def train(model,epochs,learning_rate,train_loader,valid_loader):
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    valid_loss_best = np.inf
    best_model_state = None
    count = 0
    for epoch in range(epochs):
        train_losses = []

        # train_model
        model.train()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            hypothesis = model(x)
            loss = loss_func(hypothesis,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        # validation
        valid_loss, valid_accuracy = evaluate(model,valid_loader)
        # 과적합이 일어나는 경우 중간에 학습을 종료 및 모델 정보 저장
        if valid_loss_best > valid_loss:
            valid_loss_best = valid_loss
            best_model_state = model.state_dict()
            count = 0
        else:
            if count > 3:
                torch.save(best_model_state,'checkpoint_LinearModel.pt')
                model.load_state_dict(best_model_state)
                # model.load_state_dict(torch.load('checkpoint_LinearModel.pt'))
                print('Early Stop!')
                break
            count += 1
        print(f'epoch{epoch:>{len(str(epochs))}}\ttrain_loss:{train_loss:.5f}\tvalid_loss:{valid_loss:.5f}'
              f'\taccuracy:{valid_accuracy * 100:2.2f}')



def evaluate(model,dataloader):
    loss_func = nn.CrossEntropyLoss()

    losses = []
    accuracy = []

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            out = model(x)
            loss = loss_func(out, y)
            losses.append(loss.item())

            preds = out.argmax(dim=1)
            accuracy.append((preds==y).sum().item()/len(preds))

    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracy)

    return avg_loss, avg_accuracy


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'batchsize':128,
        'inputsize':784,
        'outputsize':10,
        'dropout':0.2,
        'lr':0.0001,
        'epoch':100
    }
    print(f'using {device}')

    # 데이터셋 loader 객체로 가져오기
    train_loader, valid_loader, test_loader = prepare_datasets(config['batchsize'])

    model = MNistClassificationModel_Linear(input_size=config['inputsize'],
                                            output_size=config['outputsize'],
                                            dropout=config['dropout']).to(device)

    # 모델 학습
    train(model,epochs=config['epoch'],learning_rate=config['lr'],
          train_loader=train_loader,valid_loader=valid_loader)

    # 학습된 모델에 대한 test 수행
    test_loss, test_accuracy = evaluate(model,test_loader)
    print(f'Test Results\nLoss:{test_loss:.5f}\nAccuracy:{test_accuracy*100:2.2f}')

