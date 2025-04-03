import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


train_epochs = 20
batch_size = 100
learning_rate = 0.0002
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

dataloader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,
                        drop_last=True)

def add_noise(img):
    noise = torch.randn(img.size())*0.3
    noise_img = img + noise
    return noise_img


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3)
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

    def forward(self,x):
        eoutput = self.encoder(x)
        doutput = self.decoder(eoutput)
        return eoutput,doutput

def train(model,train_loader,loss_func,optimizer):
    avg_loss = 0.0

    for image,_ in train_loader:
        x_data = add_noise(image)
        x_data = x_data.view(-1,784).to(device)
        y = image.view(-1,784).to(device)

        _, decoded = model(x_data)
        loss = loss_func(decoded,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    return avg_loss/len(train_loader)

import numpy as np


if __name__ == '__main__':
    autoencoder = AutoEncoder().to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(train_epochs):
        loss = train(autoencoder,dataloader,loss_func,optimizer)
        print(f'epoch {epoch}\tloss:{loss:.4f}')

    sample_data = mnist_train.data[1].view(-1,784).type(torch.FloatTensor)/255.

    autoencoder.cpu()
    original_x = sample_data[0]
    noise_x = add_noise(original_x)
    _,recovered_x = autoencoder(noise_x)

    fig, axes = plt.subplots(1,3,figsize=(10,10))
    original_img = np.reshape(original_x.data.numpy(),(28,28))
    noise_img = np.reshape(noise_x.data.numpy(),(28,28))
    recovered_img = np.reshape(recovered_x.data.numpy(),(28,28))

    axes[0].set_title('original')
    axes[0].imshow(original_img,cmap='gray')
    axes[1].set_title('noise add img')
    axes[1].imshow(noise_img,cmap='gray')
    axes[2].set_title('recovered img')
    axes[2].imshow(recovered_img,cmap='gray')

    plt.show()



