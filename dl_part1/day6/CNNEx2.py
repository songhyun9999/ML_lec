import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


train_epoch = 10
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = dset.MNIST(root='MNIST_data/',
                   train=True,
                   transform=transforms.ToTensor(),
                   download=True)

mnist_test = dset.MNIST(root='MNIST_data/',
                   train=False,
                   transform=transforms.ToTensor(),
                   download=True)

data_loader = DataLoader(dataset=mnist_train,batch_size=batch_size,
                         shuffle=True,drop_last=True)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        # inputsize : batch*1*28*28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # inputsize : batch*32*14*14
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # inputsize : batch*64*7*7
        self.fc = nn.Linear(64*7*7,10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        y = self.fc(out)

        return y


if __name__ == '__main__':
    print(f'using {device}\n')
    model = CNNModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    total_batch = len(data_loader)

    for epoch in range(train_epoch):
        avg_loss = 0
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            x_train, y_train = batch

            hypothesis = model(x_train)
            loss = loss_func(hypothesis,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / total_batch
        print(f'epoch {epoch}\tavg_loss:{avg_loss:.5f}')


        with torch.no_grad():
            x_test = mnist_test.data.view(len(mnist_test),1,28,28).float().to(device)
            y_test = mnist_test.targets.to(device)
            pred = model(x_test)
            correction_prediction = torch.argmax(pred,dim=1) == y_test
            accuracy = correction_prediction.float().mean()
            print(f'Test accuracy:{accuracy*100:2.2f}')


