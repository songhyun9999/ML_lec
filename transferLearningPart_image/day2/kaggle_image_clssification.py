import os
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# 전역으로 사용할 변수 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'num_epochs':10,
    'lr':0.001,
    'batch_size':64
}
dataset_dir = 'kaggle_data'
class_names = ['buildings','forest','glacier','mountain','sea','street']

# 데이터셋 준비
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def _make_dataset(self):
        instances = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            for file_path in files:
                item = (file_path, class_idx)
                instances.append(item)
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_idx = self.samples[idx]
        image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, class_idx


# transform 정의 및 dataloader 생성
# dataloader 는 train, test 두개를 각각 생성하여 dictionary 형태로 반환
def prepare_dataset():
    train_path = os.path.join(dataset_dir, 'seg_train')
    test_path = os.path.join(dataset_dir, 'seg_test')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224,224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 학습,테스트 데이터 생성
    train_set = CustomDataset(train_path,transform=transform)
    test_set = CustomDataset(test_path, transform=transform)

    # print(len(train_set),len(test_set))

    # 데이터로더 객체 생성하여 return
    dataloaders = {
        'train':DataLoader(train_set,config['batch_size'],shuffle=True),
        'test':DataLoader(test_set,config['batch_size'],shuffle=False)
    }

    return dataloaders


# resnet18 모델 함수
def get_resnet18_model():
    # model = models.resnet18(pretrained=True)
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    # print(model)

    # pretrained model 학습 안되게 설정
    for param in model.parameters():
        param.requires_grad = False
    in_ftr = model.fc.in_features

    # fc layer 재정의
    model.fc = nn.Linear(in_ftr, len(class_names), bias=True)

    # fc layer 에 대한 학습 진행되게 설정
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# 학습 함수 정의
def train(model,dataloader):
    optimizer = optim.Adam(model.parameters(),lr=config['lr'])
    loss_func = nn.CrossEntropyLoss()

    train_losses, test_losses, train_accs, test_accs = [],[],[],[]

    # 학습 진행
    for epoch in range(config['num_epochs']):
        train_loss, test_loss = 0.0, 0.0
        model.train()
        correct,total = 0,0
        for step,batch in enumerate(dataloader['train']):
            # batch cuda 에 올리기
            images,labels = tuple(t.to(device) for t in batch)

            # model output과 loss
            out = model(images)
            loss = loss_func(out,labels)
            train_loss += loss.item()

            # 역전파 및 weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train accuracy 구하기
            correct += (out.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            # print(f'epoch {epoch}, step {step}')

        # loss 와 accuray 계산
        train_loss = train_loss / len(dataloader['train'])
        train_acc = correct/total
        test_loss, test_acc = evaluate(model,loss_func, dataloader['test'])

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'epoch {epoch}\ntrain_loss : {train_loss:.5f}\ttrain_accuracy : {train_acc:2.2f}')
        print(f'test_loss : {test_loss:.5f}\ttest_accuracy : {test_acc:2.2f}')
        print()


    # 모델 저장
    model_save_dir = 'train_results'
    os.makedirs(model_save_dir,exist_ok=True)
    model_save_path = os.path.join(model_save_dir,'kaggle_image_model.pth')

    torch.save(model.state_dict(),model_save_path)

    return train_losses,test_losses,train_accs,test_accs


# 평가 함수 정의
def evaluate(model,loss_func,dataloader):
    model.eval()
    test_loss = 0.0
    correct,total = 0,0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # batch cuda 에 올리기
            images, labels = tuple(t.to(device) for t in batch)

            # model output과 loss
            out = model(images)
            loss = loss_func(out, labels)
            test_loss += loss.item()

            correct += (out.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    test_loss = test_loss / len(dataloader)
    acc = correct/total

    return test_loss, acc


if __name__ == '__main__':
    print(f'using {device}')

    dataloader = prepare_dataset()
    model = get_resnet18_model()
    model.to(device)
    
    # 학습 진행 후 loss 및 accuracy 저장
    train_losses, test_losses, train_accs, test_accs = train(model,dataloader)
    with open("training_result.pkl", "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
        }, f)

    # 저장 결과 확인
    model_save_dir = 'train_results'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'kaggle_image_model.pth')
    model.load_state_dict(torch.load(model_save_path,map_location='cpu'))
    model.to(device)
    loss,acc = evaluate(model,nn.CrossEntropyLoss(),dataloader['test'])
    print(f'loss : {loss:.5f}\tacc : {acc:2.2f}')

    # Loss, accuracy 출력
    with open("training_result.pkl", "rb") as f:
        data = pickle.load(f)
        train_losses = data["train_losses"]
        test_losses = data['test_losses']
        train_accs = data['train_accs']
        test_accs = data['test_accs']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Loss 그리기
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(test_losses, label='Test Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy 그리기
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(test_accs, label='Test Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

