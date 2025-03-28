import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import ssl
import torch.nn.functional as F


ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(80)

def prepareDatasets():
    cifar10_train = datasets.CIFAR10('CIFAR10_data/',
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())

    cifar10_test = datasets.CIFAR10('CIFAR10_data/',
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())

    print(len(cifar10_train),len(cifar10_test))

    image, label = cifar10_train[0]
    print(image.shape)
    print(label)
    classes = cifar10_train.classes

    # plt.imshow(image.permute(1,2,0))
    # plt.show()

    cifar10_train_loader = DataLoader(cifar10_train,batch_size=1024,shuffle=True)
    cifar10_test_loader = DataLoader(cifar10_test,batch_size=2000)

    return cifar10_train_loader, cifar10_test_loader, classes

class CifarClassificationModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(CifarClassificationModel,self).__init__()
        self.fc1 = nn.Linear(input_size,16*16*3)
        self.fc2 = nn.Linear(16*16*3,16*16)
        self.fc3 = nn.Linear(16*16,8*8)
        self.fc4 = nn.Linear(8*8,output_size)

        self.dropout = 0.2

    def forward(self,x):
        x = x.view((-1,3*32*32))
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.fc3(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        y = self.fc4(out)
        return y

def train(model,epochs):
    train_loader, test_loader, classes = prepareDatasets()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        total_loss = []
        total_accuracy = []
        model.train()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            x,y = batch

            hypothesis = model(x)
            loss = loss_func(hypothesis,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            acc = (hypothesis.argmax(dim=1) == y).sum().item() / len(hypothesis)
            total_accuracy.append(acc)

        total_loss = np.mean(total_loss)
        total_accuracy = np.mean(total_accuracy)
        val_loss, val_accuracy = evaluate(model,test_loader)
        print(f'epoch {epoch}\ttrain_loss:{total_loss:.5f}'
              f'\tvalid_loss:{val_loss:.5f}'
              f'\ttrain_accuray:{total_accuracy*100:2.2f}'
              f'\tvalid_accuracy:{val_accuracy*100:2.2f}')

        train_losses.append(total_loss)
        test_losses.append(val_loss)
        train_accuracy.append(total_accuracy)
        test_accuracy.append(val_accuracy)

    return train_losses,test_losses,train_accuracy,test_accuracy

def evaluate(model,dataloader):
    loss_func = nn.CrossEntropyLoss()
    total_loss = []
    total_accuracy = []
    model.eval()
    with torch.no_grad():
        for step,batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch

            preds = model(x)
            loss = loss_func(preds, y)
            total_loss.append(loss.item())

            acc = (preds.argmax(dim=1) == y).sum().item() / len(preds)
            total_accuracy.append(acc)

        total_loss = np.mean(total_loss)
        acc = np.mean(total_accuracy)

    return total_loss, acc




if __name__ == '__main__':
    print(f'using {device}')

    epochs = 20
    model = CifarClassificationModel(3*32*32,10).to(device)
    train_losses, test_losses, train_accuracy, test_accuracy= train(model,epochs=epochs)

    with torch.no_grad():
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(train_losses,label='training loss')
        plt.plot(test_losses,label='test loss')
        plt.legend(loc='best')

        plt.subplot(122)
        plt.plot(train_accuracy, label='training accuracy')
        plt.plot(test_accuracy, label='test accuracy')
        plt.legend(loc='best')

    plt.show()