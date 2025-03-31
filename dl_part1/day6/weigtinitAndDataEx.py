import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init


batch_size = 100
learning_rate = 0.001
n_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fmnist_train = dset.FashionMNIST(root='FashionMNIST_data/',
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(34),
                                     transforms.CenterCrop(28),
                                     transforms.Lambda(lambda x:x.rotate(90)),
                                     transforms.ToTensor()
                                 ]),
                                 download=True,
                                 target_transform=None
                                 )

fmnist_test = dset.FashionMNIST(root='FashionMNIST_data/',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                target_transform=None,
                                download=True)

train_loader = DataLoader(fmnist_train,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(fmnist_test,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=True)

class CNNet2(nn.Module):
    def __init__(self):
        super(CNNet2,self).__init__()
        self.Clayer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fclayer = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.Linear):
                init.xavier_normal_(m.weight.data)


    def forward(self,x):
        out = self.Clayer(x)
        out = out.view(out.size(0),-1)
        y = self.fclayer(out)

        return y

model = CNNet2().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)

for epoch in range(n_epochs):
    total_loss = 0.0
    for x_train, y_train in train_loader:
        x_train,y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis,y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'epoch:{epoch}\tloss:{total_loss:.4f}')