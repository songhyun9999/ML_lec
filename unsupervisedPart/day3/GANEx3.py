import glob
import os.path

from PIL import Image
import matplotlib.pyplot as plt

# path_to_imgs = './GANIMG/img_align_celeba'
# imgs = glob.glob(os.path.join(path_to_imgs,'*'))
# print(imgs)
#
# for i in range(9):
#     plt.subplot(3,3,1+i)
#     img = Image.open(imgs[i])
#     plt.imshow(img)
#
# plt.show()

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# config
batch_size = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0002
num_epoch = 50


# Datasets
dataset = ImageFolder(
    root='./GANIMG',
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
)

dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

# it = iter(dataloader)
# data = it.__next__()
# print(data)
# print(data[0].shape)

# model
# torch.Size([128, 3, 64, 64])
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # 64x32x32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 128x16x16
            nn.Conv2d(128, 256, 3, padding=1),  # 256x7x7
            nn.LeakyReLU(0.2), # 256x16x16
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 256x8x8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # 512x4x4
            nn.Conv2d(512, 512, 3, padding=1),  # 256x7x7
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1,512*4*4)
        y = self.fc(out)

        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3,
                               2, 1, 1),  # 256x8x8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3,
                               1, 1),  # 128x8x8
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3,
                               2, 1, 1),  # 64x16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3,
                               1, 1),  # 64x16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3,
                               2, 1,1),  # 32x32x32
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3,
                               2, 1, 1),  # 3x64x64
            nn.Tanh() # -1 ~ 1
        )

    def forward(self,x):
        x = x.view(-1,512,4,4)
        out = self.layer1(x)
        out = self.layer2(out)
        y = self.layer3(out)

        return y

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)


if __name__ == '__main__':
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    discriminator.apply(weight_init)
    generator.apply(weight_init)

    loss_func = nn.BCELoss()
    # betas option 의 인자 : beta1, beta2 -> beta1 줄이면 기울기 반영률 증가
    d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate)#,betas=(0.5,0.999))
    g_optimizer = optim.Adam(generator.parameters(),lr=learning_rate)#,betas=(0.5,0.999))

    # fake image 512*4*4
    for epoch in range(num_epoch):
        for step,(images,_) in enumerate(dataloader):
            images = images.to(device)
            real_label = torch.ones(images.size(0), 1).to(device)
            fake_label = torch.zeros(images.size(0), 1).to(device)

            # noise 생성
            z = torch.randn(images.size(0),512*4*4).to(device)
            fake_images = generator(z)
            # real image 에 대한 판별자 loss
            outputs = discriminator(images)
            d_loss_real = loss_func(outputs,real_label)

            # fake image 에 대한 판별자 loss
            outputs = discriminator(fake_images.detach())
            d_loss_fake = loss_func(outputs,fake_label)

            # 판별자 학습
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 생성자 학습
            z = torch.randn(images.size(0), 512 * 4 * 4).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = loss_func(outputs,real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if step%10 == 0:
                print(f'epoch {epoch} step {step}\td_loss:{d_loss.item():.4f},g_loss:{g_loss.item():.4f}')

    z = torch.randn(batch_size, 512*4*4)
    generator.cpu()
    fake_images = generator(z)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        fake_img = fake_images[i].detach().cpu().numpy()
        fake_image_img = np.transpose(fake_img, (1, 2, 0))
        fake_image_img = (fake_image_img + 1) / 2
        plt.imshow(fake_image_img)

    plt.show()