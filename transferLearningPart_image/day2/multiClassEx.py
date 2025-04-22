import os
import json

'''
    어노테이션 파일을 불러온다.
'''

data_root = 'data' # 데이터가 저장되어 있는 루트 경로
annotation_filename = 'annotations.json' # 어노테이션 파일의 이름

# json으로 데이터를 불러온다.
with open(os.path.join(data_root, annotation_filename), 'r')as json_f:
    annotations = json.load(json_f)

print(annotations[:3]) # 데이터의 샘플을 출력한다.

from collections import defaultdict

'''
    데이터의 기본 속성을 파악한다. (데이터 개수, 클래스 개수, 클래스 개수 분포 등)
'''


# 클래스맵을 생성하는 함수
def get_class_map(annotations):
    cls_map = defaultdict(int)
    for annot in annotations:
        for cls in annot['classes']:
            cls_map[cls] += 1

    return cls_map

cls_map = get_class_map(annotations)
print(cls_map)
print(f'데이터 개수 : {len(annotations)}')
print(f'클래스 개수 : {len(list(cls_map.keys()))}')

import matplotlib.pyplot as plt

'''
    이미지 시각화 함수를 정의한다.
'''


def draw_images(images, classes):
    '''
        :param images: cv2(ndarray) 이미지 리스트
        :param classes: 클래스 리스트
        :return: None
    '''
    # 4x2의 그리드 생성 (바둑판 이미지 틀 생성)
    fig, axs = plt.subplots(2, 4)

    # 각 하위 그래프에 이미지 출력
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])  # 이미지를 바둑판에 출력
        ax.set_title(classes[i])  # 클래스 이름으로 이미지 제목 생성
        ax.axis('off') # 축 숨기기 (이미지 크기 출력 해제)

    plt.tight_layout()
    plt.show()


from PIL import Image
import numpy as np
import random

'''
    데이터를 랜덤하게 셔플하고 시각화를 수행한다.
'''

random.shuffle(annotations)
sample_images = [] # 이미지 샘플 저장
sample_classes = [] # 이미지 클래스 저장
sample_cnt = 0 # 시작 count
max_cnt = 8 # 종료 count


'''
    어노테이션을 탐색하여 클래스와 파일경로 정보를 불러온다.
    데이터 구조는 아래와 같다.
    [
        {
            'filepath': 'green_shirt/915c509883a9b65d343fb7411c973f11d953bd04.jpg',
            'filename': '915c509883a9b65d343fb7411c973f11d953bd04.jpg',
            'classes': ['green', 'shirt']
        },
        ...
    ]
'''

for annot in annotations:
    sample_classes.append(annot['classes'])
    image_path = os.path.join(data_root,annot['filepath'])
    image = Image.open(image_path).convert('RGB')
    sample_images.append(np.array(image))
    sample_cnt += 1
    if sample_cnt == max_cnt:
        break

## 전체 데이터 샘플을 시각화 한다.
# draw_images(sample_images,sample_classes)

from PIL import Image
import numpy as np
import random

'''
    클래스별 데이터를 시각화 한다.
'''
sample_images = [] # 이미지 샘플 저장
sample_classes = [] # 이미지 클래스 저장
sample_cnt = 0 # 시작 count
max_cnt = 8 # 종료 count

# white, shoes 에 대한 데이터 시각화
for annot in annotations:
    if 'white' not in annot['classes'] or 'shoes' not in annot['classes']:
        continue
    sample_classes.append(annot['classes'])
    image_path = os.path.join(data_root,annot['filepath'])
    image = Image.open(image_path).convert('RGB')
    sample_images.append(np.array(image))
    sample_cnt += 1
    if sample_cnt == max_cnt:
        break

# draw_images(sample_images,sample_classes)


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

'''
    json 데이터를 파싱하는 커스텀 데이터세트 클래스를 선언한다.
'''


class JsonDataset(Dataset):
    def __init__(self,
                 data_root,
                 annotations,
                 transform=None):
        '''
            :param data_root: 데이터셋의 루트 경로
            :param annotations: 어노테이션
            :param transform: 이미지 변환 모듈
        '''
        self.data_root = data_root
        self.annotations = annotations
        self.transform = transform
        self.class_list = self._get_classes()  # 클래스의 목록
        self.num_classes = len(self.class_list)  # 클래스 개수

    def __len__(self):
        return len(self.annotations)  # 데이터 개수

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        image_path = os.path.join(self.data_root,annot['filepath'])
        image = Image.open(image_path).convert('RGB')
        classes = annot['classes']

        # one-hot vector 생성
        target = []
        for cls in classes:
            target.append(self.class_list.index(cls))
        target = F.one_hot(torch.tensor(target),self.num_classes).sum(dim=0).to(torch.float)

        if self.transform:
            image = self.transform(image)

        return image,target


    def _get_classes(self):
        class_set = set()
        for annot in annotations:
            for cls in annot['classes']:
                class_set.add(cls)
        class_list = list(class_set)
        class_list.sort()

        return class_list

dataset = JsonDataset(data_root=data_root,annotations=annotations)
data = dataset[0]
print(data[1])
print(data[0])


'''
    데이터세트를 학습과 검증셋으로 분리한다.
'''

random.shuffle(annotations)
len_annot = len(annotations)
train_annot = annotations[:int(len_annot*0.9)]
val_annot = annotations[int(len_annot*0.9):]

print(f'학습 데이터 개수 : {len(train_annot)}')
print(f'검증 데이터 개수 : {len(val_annot)}')


'''
    학습에 필요한 하이퍼파라미터를 선언한다.
'''

hyper_params = {
    'num_epochs':5,
    'lr':0.0001,
    'score_threshold':0.5, # 모델 출력값에 대한 임계값
    'image_size':224,
    'train_batch_size':8,
    'val_batch_size':4,
    'print_preq':0.1 # 학습 중 로그 출력 빈도
}


from torchvision import transforms

'''
    이미지 변환 모듈을 적용한 데이터세트의 결과물을 확인한다.
'''
# 샘플 이미지 변환 모듈 설정
sample_transform = transforms.Compose([
    transforms.Resize((hyper_params['image_size'],hyper_params['image_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # transforms.ToTensor() # 테스트시에 사용X
])
sample_dataset = JsonDataset(data_root,train_annot,sample_transform)

transformed_images = []
targets = []

# 데이터셋에서 변환된 이미지와 target 벡터를 불러온다.
max_cnt = 8
for idx,(image,target) in enumerate(sample_dataset):
    if idx == max_cnt:
        break
    transformed_images.append(image)
    targets.append(target.tolist())

## 타겟 벡터를 클래스로 변환한다.
target_classes = []
class_list = sample_dataset.class_list
for target in targets:
    classes = []
    for cls,val in enumerate(target):
        if int(val) == 1:
            classes.append(class_list[cls])
    target_classes.append(classes)

draw_images(transformed_images,target_classes)

'''
    이미지 변환 모듈을 적용한 학습 및 검증 데이터세트를 생성한다.
    학습 및 검증 데이터 로더를 생성한다.
'''

# 학습 및 검증 이미지 변환 모듈 설정
transform = transforms.Compose([
    transforms.Resize((hyper_params['image_size'], hyper_params['image_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])

# 학습 데이터세트 및 데이터로더 설정
train_dataset = JsonDataset(data_root, train_annot, transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyper_params['train_batch_size'], shuffle=True)

# 검증 데이터세트 및 데이터로더 설정
val_dataset = JsonDataset(data_root, val_annot, transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hyper_params['val_batch_size'], shuffle=False)


import torch
import torch.nn as nn
from torchvision import models

'''
    1. VGG16 모델을 불러온다.
    2. 클래스 개수에 맞게 출력 레이어를 변경한다.
'''
# import ssl
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# model = models.vgg16(pretrained = True)
# print(model)
# print()
#
# ### 에러 발생시
# # model = models.vgg16(weights = 'VGG16_Weights.IMAGENET1K_V1')
#
# model.classifier[-1] = nn.Linear(4096,train_dataset.num_classes,bias=True)
# print(model)
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import f1_score
#
# # loss 함수와 옵티마이저 설정
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(),lr=hyper_params['lr'])
#
# # 장치 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# num_epochs = hyper_params['num_epochs']
#
# # 학습 루프
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     epoch_loss = 0.0
#     print_cnt = int(len(train_dataloader) * hyper_params['print_preq'])
#
#     for idx, (images, targets) in enumerate(train_dataloader):
#         images, targets = images.to(device), targets.to(device)
#
#         # forward
#         outputs = model(images)
#         loss = criterion(outputs,targets)
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         epoch_loss += loss.item()
#
#         if idx % print_cnt == 0:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], "
#                   f"Iter [{idx}/{len(train_dataloader)}] "
#                   f"Loss: {running_loss / print_cnt:.4f}")
#             running_loss = 0.0
#
#     # 한 epoch이 끝날 때마다 loss 총 합 출력
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")
#
#     # F1-score 계산 및 출력
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for images, targets in val_dataloader:
#             images = images.to(device)
#             outputs = torch.sigmoid(model(images))  # 시그모이드로 출력값 변환
#             preds = (outputs > hyper_params['score_threshold']).float()  # threshold 설정 (정답값 선택)
#             y_true.extend(targets.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())
#
#     f1 = f1_score(y_true, y_pred, average='micro')  # F1-score 계산
#     print(f"F1-score: {f1:.4f}")
#
# '''
# #    학습 결과 모델과 하이퍼파라미터를 저장한다.
# '''
# model_save_dir = 'train_results'
# os.makedirs(model_save_dir,exist_ok=True)
# model_save_path = os.path.join(model_save_dir,'model.pth')
#
# torch.save(model.state_dict(),model_save_path)
#
# param_save_path = os.path.join(model_save_dir,'hyper_params.json')
# with open(param_save_path,'w') as json_f:
#     json.dump(hyper_params,json_f,indent='\t',ensure_ascii=False)


'''
#    모델을 선언하고 학습한 모델의 가중치를 불러온다.
'''

model = models.vgg16(pretrained = True)
model.classifier[-1] = nn.Linear(4096,train_dataset.num_classes,bias=True)
model_save_dir = 'train_results'
model_save_path = os.path.join(model_save_dir,'model.pth')
model.load_state_dict(torch.load(model_save_path,map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

'''
#    모델의 추론을 수행한다.
'''
image_list = []
pred_list = []

val_cnt = 8
with torch.no_grad():
    model.eval()
    for idx,annot in enumerate(val_annot):
        if idx == val_cnt:
            break
        image_path = os.path.join(data_root,annot['filepath'])
        print(f'image_path : {image_path}')
        image = Image.open(image_path)
        image_list.append(image)
        input_image = transform(image).unsqueeze(0).to(device)

        outputs = torch.sigmoid(model(input_image))[0]
        preds = (outputs > hyper_params['score_threshold']).int().tolist() # threshold 설정
        pred_list.append(preds)

'''
#    모델의 추론 결과(클래스 번호)를 클래스 이름으로 변경한다.
'''
class_list = train_dataset.class_list
pred_class_list = []
for pred in pred_list:
    pred_class = []
    for cls,val in enumerate(pred):
        if int(val)==1:
            pred_class.append(class_list[cls])
    pred_class_list.append(pred_class)

draw_images(image_list,pred_class_list)

