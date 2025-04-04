from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

root = './images'
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=root,transform=transform)
dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)

print(dataset.classes)

images, labels = next(iter(dataloader))
print(images.shape)
print(labels.shape)
print(labels)

labels_map = {v:k for k,v in dataset.class_to_idx.items()}
print(labels_map)
print(len(dataset))

from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset,[train_size,test_size])
batch_size = 32

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)


from torchvision import models
from torchvision.models import VGG16_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = models.vgg16(weights=VGG16_Weights.DEFAULT)
# print(model)

for param in model.parameters():
    param.requires_grad = False

fc = nn.Sequential(
    nn.Linear(512*7*7,256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,2)
)

model.classifier = fc
print(model)
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=0.00005)
loss_func = nn.CrossEntropyLoss()


def train(model,dataloader,loss_func,optimizer):
    model.train()
    running_loss = 0.0
    corr = 0

    for step,batch in enumerate(dataloader):
        x_train, y_train = tuple(t.to(device) for t in batch)

        hypothesis = model(x_train)
        loss = loss_func(hypothesis,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = hypothesis.argmax(dim=1)
        corr += pred.eq(y_train).sum().item()

        running_loss += loss.item() * x_train.size(0)
        print(f'step {step}/{len(dataloader)}')

    acc = corr / len(dataloader.dataset)
    return running_loss/len(dataloader.dataset), acc

def evaluate(model,dataloader,loss_func):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0.0
        for x_test,y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            out = model(x_test)
            pred = out.argmax(dim=1)
            corr += torch.sum(pred.eq(y_test)).item()
            running_loss += loss_func(out,y_test).item() * x_test.size(0)

        acc = corr / len(dataloader.dataset)
        return running_loss/len(dataloader.dataset), acc

num_epoch = 5
model_name = 'vgg16-trained'
min_loss = np.inf

for epoch in range(num_epoch):
    train_loss, train_acc = train(model,train_loader,loss_func, optimizer)
    val_loss, val_acc = evaluate(model,test_loader,loss_func)

    if val_loss < min_loss:
        print(f'val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}, saving model!!!')
        min_loss = val_loss
        torch.save(model.state_dict(),f'{model_name}.pth')

    print(f'epoch:{epoch}\tloss:{train_loss:.5f}\tacc:{train_acc*100:2.3f}'
          f'\tval_loss:{val_loss:.5f}\tval_acc:{val_acc*100:2.3f}')

model.load_state_dict(torch.load(f'{model_name}.pth'))
final_loss, final_acc = evaluate(model,test_loader,loss_func)
print(f'evaluation loss:{final_loss:.5f}\tevaluation accuracy:{final_acc*100:2.3f}')

