import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

torch.manual_seed(12345)
# -------------------------
# 1. Device & Config
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 500
train_epoch = 30

# -------------------------
# 2. Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------
# 3. Dataset & Loader
# -------------------------
cifar10_train = dset.CIFAR10(root='CIFAR10_data', train=True, download=True, transform=train_transform)
cifar10_test = dset.CIFAR10(root='CIFAR10_data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

# -------------------------
# 4. Model Definition
# -------------------------
class BetterCifar10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        def conv_block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.layer1 = conv_block(3, 64)
        self.layer2 = nn.Sequential(
            conv_block(64, 128, stride=2),
            conv_block(128, 128)
        )
        self.layer3 = nn.Sequential(
            conv_block(128, 256, stride=2),
            conv_block(256, 256)
        )
        self.layer4 = nn.Sequential(
            conv_block(256, 512, stride=2),
            conv_block(512, 512)
        )

        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# -------------------------
# 5. Evaluation
# -------------------------
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            pred = model(x_test)
            correct += (pred.argmax(1) == y_test).sum().item()
            total += y_test.size(0)
    return correct / total

# -------------------------
# 6. Training
# -------------------------
def train(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_epoch)

    best_acc = 0.0
    early_stop_counter = 0
    max_patience = 10

    for epoch in range(train_epoch):
        model.train()
        total_loss = 0.0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            pred = model(x_train)
            loss = loss_func(pred, y_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        acc = evaluate(model)

        # Early stopping & best model save
        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_cifar10_model.pt')
        else:
            early_stop_counter += 1
            if early_stop_counter >= max_patience:
                print("Early stopping triggered.")
                break

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc*100:.2f}%")

    print(f"Best Accuracy: {best_acc*100:.2f}%")

# -------------------------
# 7. Run
# -------------------------
if __name__ == '__main__':
    print(f"Using device: {device}")
    model = BetterCifar10CNN().to(device)
    train(model)
    # Load best model
    model.load_state_dict(torch.load('best_cifar10_model.pt'))
    final_acc = evaluate(model)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
