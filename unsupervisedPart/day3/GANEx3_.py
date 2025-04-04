import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim

# path_to_imgs = './GANIMG/img_align_celeba'
# imgs = glob.glob(os.path.join(path_to_imgs, '*'))
# print(imgs)
#
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     img = Image.open(imgs[i])
#     plt.imshow(img)
#
# plt.show()

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = ImageFolder(
    root='./GANIMG',
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
)

data_loader = DataLoader(dataset=dataset,
                         batch_size=128,
                         shuffle=True)
# it = iter(data_loader)
# data = next(it)
# print(data)
# print(data[0].shape)



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.gen(x)
        return y

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.disc(x)
        return y

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(generator,discriminator):
    G_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    for epoch in range(50):
        for step,data in enumerate(data_loader):
            data = tuple(t.to(device)for t in data)
            D_optimizer.zero_grad()

            label_real = torch.ones_like(data[1], dtype=torch.float32)
            label_fake = torch.zeros_like(data[1], dtype=torch.float32)

            hypothesis = discriminator(data[0])

            d_loss_real = nn.BCELoss()(torch.squeeze(hypothesis), label_real)
            d_loss_real.backward()

            noise = torch.randn(label_real.shape[0], 100, 1, 1).to(device)
            fake_img = generator(noise)

            hypothesis2 = discriminator(fake_img.detach())

            d_loss_fake = nn.BCELoss()(torch.squeeze(hypothesis2), label_fake)
            d_loss_fake.backward()
            D_optimizer.step()

            d_loss = d_loss_real + d_loss_fake

            G_optimizer.zero_grad()
            hypothesis3 = discriminator(fake_img)
            g_loss = nn.BCELoss()(torch.squeeze(hypothesis3), label_real)
            g_loss.backward()
            G_optimizer.step()

            if step % 10 ==0:
                print(f'epoch:{epoch} d_loss:{d_loss.item():4f} g_loss:{g_loss.item():4f}')

    torch.save(G.state_dict(), f'Generator_best.pth')
    torch.save(D.state_dict(), f'Discriminator_best.pth')

if __name__ == '__main__':
    print(f'using {device}')
    G = Generator()
    G.apply(weight_init)

    D = Discriminator()
    D.apply(weight_init)

    if os.path.exists('./Generator_best.pth'):
        G.load_state_dict(torch.load('./Generator_best.pth'))

    if os.path.exists('./Discriminator_best.pth'):
        D.load_state_dict(torch.load('./Discriminator_best.pth'))

    G.to(device)
    D.to(device)

    train(G,D)

    noise = torch.randn(64, 100, 1, 1)

    with torch.no_grad():
        G.cpu()
        G.load_state_dict(torch.load('./Generator_best.pth'))
        noise = torch.randn(1, 100, 1, 1)
        pred = G(noise).squeeze()
        pred = pred.permute(1, 2, 0).numpy()
        pred = (pred+1.0)/2.0

        plt.imshow(pred)
        plt.title('prediction image')
        plt.show()

