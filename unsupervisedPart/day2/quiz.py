import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

train_epochs = 6
batch_size = 100
learning_rate = 0.0002
input_size=784
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fmnist_train = dset.FashionMNIST(root='FashionMNIST_data/',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)

fmnist_test = dset.FashionMNIST(root='FashionMNIST_data/',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)

dataloader = DataLoader(fmnist_train,batch_size=batch_size,shuffle=True)

# 100x1x28x28

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,30)
        )

        self.decoder = nn.Sequential(
            nn.Linear(30,100),
            nn.ReLU(),
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,input_size),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.view(batch_size,-1)
        out = self.encoder(x)
        y = self.decoder(out).view(batch_size,1,28,28)
        return y


if __name__ == '__main__':
    AEmodel = AutoEncoder().to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(AEmodel.parameters(),lr=learning_rate)

    for epoch in range(train_epochs):
        for step,batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            x_train, _ = batch

            hypothesis = AEmodel(x_train)
            loss = loss_func(hypothesis,x_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f'step {step}\tloss:{loss.item():.4f}')

    plt.figure(figsize=(8,6))
    for i in range(5):
        plt.subplot(2,5,1+i)
        plt.imshow(torch.squeeze(x_train[i]).cpu().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(2,5,6+i)
        plt.imshow(torch.squeeze(hypothesis.data)[i].cpu().numpy(),cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
