import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


train_epochs = 100
batch_size = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

mnist_train = dset.FashionMNIST(root='FashionMNIST_data/',
                         train=True,
                         transform=transform,
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True)


Generator = nn.Sequential(
    nn.Linear(64,256),
    nn.ReLU(),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Linear(256,784),
    nn.Tanh() # 출력 범위 확장 -> -1~1  (sigmoid : 0~1)
).to(device)

Discriminator = nn.Sequential(
    nn.Linear(784,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
).to(device)

loss_func = nn.BCELoss()
d_optimizer = optim.Adam(Discriminator.parameters(),lr=0.0002)
g_optimizer = optim.Adam(Generator.parameters(),lr=0.0002)

print(f'using {device}')

for epoch in range(train_epochs):
    for image,_ in data_loader:
        image = image.view(batch_size,-1).to(device)
        real_label = torch.ones(batch_size,1).to(device)
        fake_label = torch.zeros(batch_size,1).to(device)

        # noise 에 대한 이미지 생성
        z = torch.randn(batch_size,64).to(device)
        fake_images = Generator(z)

        # real image 에 대한 판별자 loss
        outputs = Discriminator(image)
        d_loss_real = loss_func(outputs, real_label)

        # fake image 에 대한 판별자 loss
        outputs = Discriminator(fake_images.detach())
        d_loss_fake = loss_func(outputs,fake_label)

        # 판별자 학습 진행
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 생성자 학습 진행
        fake_images = Generator(z)
        outputs = Discriminator(fake_images)
        g_loss = loss_func(outputs,real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f'epoch {epoch}\td_loss:{d_loss.item():.4f}\tg_loss:{g_loss.item():.4f}')


z = torch.randn(batch_size, 64)
Generator.cpu()
fake_images = Generator(z)


for i in range(3):
    plt.subplot(1,3,i+1)
    fake_image_img = np.reshape(fake_images.data.numpy()[i],(28,28))
    plt.imshow(fake_image_img,cmap='gray')

plt.show()