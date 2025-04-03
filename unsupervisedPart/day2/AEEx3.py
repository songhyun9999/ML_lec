import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

train_epochs = 10
batch_size = 64
learning_rate = 0.005

mnist_train = dset.FashionMNIST(root='FashionMNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_test = dset.FashionMNIST(root='FashionMNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        eoutput = self.encoder(x)
        doutput = self.decoder(eoutput)
        return eoutput, doutput

autoencoder = AutoEncoder()
loss_func = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

def train(autoencoder, train_loader):
    for x_data, _ in train_loader:
        x_data = x_data.view(-1, 784)

        eoutput, doutput = autoencoder(x_data)
        loss = loss_func(doutput, x_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

view_data = mnist_train.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor) / 255.

import numpy as np

for epoch in range(train_epochs):
    train(autoencoder, data_loader)

    _, doutput = autoencoder(view_data)

    print(f'epoch:{epoch+1}')
    fig, axes = plt.subplots(2,5, figsize=(5,2))
    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i], (28,28))
        axes[0][i].imshow(img, cmap='gray')
        axes[0][i].set_xticks(())
        axes[0][i].set_yticks(())

    for i in range(5):
        img = np.reshape(doutput.data.numpy()[i], (28,28))
        axes[1][i].imshow(img, cmap='gray')
        axes[1][i].set_xticks(())
        axes[1][i].set_yticks(())
    plt.show()


