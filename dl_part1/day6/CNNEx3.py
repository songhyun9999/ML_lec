import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


torch.manual_seed(12345)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1000
train_epoch = 30
# 0~1 값을 -1~1 값으로 scaling
# 맞춤형으로 적용하긴 위해선 데이터의 mean,std 값을 구해야함
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

cifar10_train = dset.CIFAR10(root='CIFAR10_data',
                                  train=True,
                                  download=True,
                                  transform=transform)

cifar10_test = dset.CIFAR10(root='CIFAR10_data',
                                  train=False,
                                  download=True,
                                  transform=transform)

# print(cifar10_train[0][0].shape)
# batchsize * 3 * 32 * 32
# print(cifar10_test[0])
train_loader = DataLoader(dataset=cifar10_train,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=cifar10_test,batch_size=batch_size)

class CNNCifar10Model(nn.Module):
    def __init__(self):
        super(CNNCifar10Model,self).__init__()
        # 3*32*32
        self.layer1 = self.make_layer(3,32,5)
        # 32*16*16
        self.layer2 = self.make_layer(32,64,5)
        # 64*8*8
        self.layer3 = self.make_layer(64,128,5)
        # 128*4*4
        self.layer4 = self.make_layer(128,128,3,False)
        # 128*4*4
        self.fc = nn.Linear(128*4*4,10)

    def make_layer(self,in_channel,out_channel,kernel_size=3,pool=True):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=1))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU())
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=1))
        layers.append(nn.Dropout(0.25))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # batch * -1
        out = out.view(out.size(0),-1)
        y = self.fc(out)

        return y

def train(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(train_epoch):
        total_loss = 0.0
        model.train()
        for step,batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            x_train, y_train = batch

            hypothesis = model(x_train)
            loss = loss_func(hypothesis,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        accuracy = evaluate(model)
        print(f'epoch {epoch}\ttotal_loss:{total_loss:.5f}\taccuracy:{accuracy*100:2.2f}')


def evaluate(model):
    # batch*3*32*32
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # N*32*32*3 -> reshape으론 틀린 결과물 나옴, permute로 차원 변경 진행
        # 하지만 이경우 이미지가 raw 상태로 transform을 거치지않아 제대로 작동 X
        # print(cifar10_test.data.shape)
        # x_test = torch.tensor(cifar10_test.data).permute(0,3,1,2).float().to(device)
        # y_test = torch.tensor(cifar10_test.targets).float().to(device)
        for step,batch in enumerate(test_loader):
            batch = tuple(t.to(device) for t in batch)
            x_test,y_test = batch
            pred = model(x_test)

            correct += (pred.argmax(1) == y_test).sum().item()
            total += y_test.size(0)

    return correct/total

if __name__ == '__main__':
    print(f'using {device}\n')

    cnn_model = CNNCifar10Model().to(device)
    train(cnn_model)



