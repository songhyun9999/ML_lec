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
input_size=784

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


dataloader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self,input_size):
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 20)
        )

        self.decoder = nn.Sequential(
            nn.Linear(20,256),
            nn.ReLU(),
            nn.Linear(256,input_size)
        )


    def forward(self,x):
        x = x.view(batch_size,-1)
        out = self.encoder(x)
        y = self.decoder(out).view(batch_size,1,28,28)

        return y

AEModel = AutoEncoder(input_size)
loss_func = nn.MSELoss()
optimizer = optim.Adam(AEModel.parameters(),lr=learning_rate)

for epoch in range(train_epochs):
    for step,(x_data,_) in enumerate(dataloader):
        hypothesis = AEModel(x_data)
        loss = loss_func(hypothesis,x_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Loss : {loss.item():.4f}')


out_img = torch.squeeze(hypothesis.data)

for i in range(3):
    plt.subplot(121)
    plt.imshow(torch.squeeze(x_data[i]).numpy(),cmap='gray')
    plt.subplot(122)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()

