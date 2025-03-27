import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

datasets.FashionMNIST
train_set = datasets.FashionMNIST(root='FashionMNIST_data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()])
)

test_set = datasets.FashionMNIST(root='FashionMNIST_data/',
                           train=False,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()])
)

from torch.utils.data import DataLoader

batch_size = 100
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(test_set,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

import torch.nn as nn
import torch.nn.functional as F

class ImageNN(nn.Module):
    def __init__(self, drop_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout_p = drop_p

    def forward(self, x):
        x = x.view(-1, 784)
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        y = self.fc3(out)
        return y


import torch.optim as optim
model = ImageNN(drop_p=0.2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for x_train, y_train in train_loader:
        #x_train = x_train.view(-1, 28*28)
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            hypothesis = model(x_test)
            test_loss += F.cross_entropy(hypothesis, y_test).item()
            pred = torch.argmax(hypothesis, dim=1)
            correct += pred.eq(y_test.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, test_accuracy

for epoch in range(20):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f'epoch:{epoch+1}, loss:{test_loss:.4f}, accuracy:{test_accuracy:2.2f}%')


















