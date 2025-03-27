import numpy as np
from torchvision import datasets
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_set = datasets.MNIST(root='MNIST_data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()])
)

test_set = datasets.MNIST(root='MNIST_data/',
                           train=False,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()])
)

from torch.utils.data import DataLoader

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

dataiter = iter(train_loader)
next(dataiter)
images, labels = next(dataiter)
print(images.shape)
print(labels)
print(labels.shape)

from torchvision import utils

img = utils.make_grid(images)
npimg = img.numpy()
print(npimg.shape)
print(np.transpose(npimg, (1,2,0)).shape)
import matplotlib.pyplot as plt

# plt.figure(figsize=(10,7))
# plt.imshow(np.transpose(npimg, (1,2,0)))
# print(labels)
# plt.show()

one_img = images[1]
print(one_img.shape)
one_npimg = one_img.squeeze().numpy()
plt.title(f'"{labels[1]}" image')
plt.imshow(one_npimg, cmap='gray')
plt.show()












