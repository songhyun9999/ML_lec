import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


train_epochs = 3
batch_size = 100
learning_rate = 0.0002
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


dataloader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,
                        drop_last=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2) # 64x14x14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2), # 128x7x7
            nn.Conv2d(128,256,3,padding=1),# 256x7x7
            nn.ReLU()
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        y = out.view(batch_size,-1)
        return y


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,
                               2,1,1), # 128x14x14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,
                               1,1), # 64x14x14
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64,16,3,
                               1,1), # 16x14x14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16,1,3,
                               2,1,1), # 1x28x28
            nn.ReLU()
        )

    def forward(self,x):
        x = x.view(batch_size,256,7,7)
        out = self.layer1(x)
        y = self.layer2(out)
        return y


encoder = Encoder().to(device)
decoder = Decoder().to(device)
loss_func = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(parameters,lr=learning_rate)

for epoch in range(train_epochs):
    for step,batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        x_train, _ = batch

        eoutput = encoder(x_train)
        hypothesis = decoder(eoutput)

        loss = loss_func(hypothesis,x_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100 == 0:
            print(f'epoch {epoch} / step {step} loss:{loss.item():.4f}')


out_img = torch.squeeze(hypothesis.data)
for i in range(3):
    plt.subplot(121)
    plt.imshow(torch.squeeze(x_train[i]).cpu().numpy(),cmap='gray')
    plt.subplot(122)
    plt.imshow(out_img[i].cpu().numpy(),cmap='gray')
    plt.show()
