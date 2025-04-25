import pandas as pd

from sklearn.model_selection import train_test_split
from accelerate import Accelerator #pip install accelerate

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler


def get_device():
  device="cpu"
  if torch.cuda.is_available():
    device="cuda"
  else:
    device="cpu"
  return device


device = get_device()
print(device)

real = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

real = real.drop(['title','subject','date'],axis=1)
real['label'] = 1.0
fake = fake.drop(['title','subject','date'],axis=1)
fake['label'] = 0.0
dataframe = pd.concat([real,fake],axis=0,ignore_index=True)

df = dataframe.sample(frac=0.1).reset_index(drop=True)
print(df.head(20))
print(len(df[df['label']==1.0])) # real
print(len(df[df['label']==0.0])) # fake



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# (text, label) 형태의 튜플로 구성된 리스트 생성
data = list(zip(df['text'].tolist(),df['label'].tolist()))

# 다음 함수는 파라미터로 texts와 lables로 구성된 리스트를 가지며
# 출력으로 input_ids, attention_mask, labels_out을 생성
def tokenize_and_encode(texts, labels):
    input_ids, attention_masks, labels_out = [],[],[]
    for text,label in zip(texts,labels):
        encoded = tokenizer.encode_plus(text,max_length=512,padding='max_length',truncation=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels_out.append(label)

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels_out)


# 튜플을 분리하여 containing texts, containing labels 리스트 생성
texts,labels = zip(*data)

# 학습 및 검증 데이터셋 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(texts,labels,test_size=0.2)

# 토큰화
train_input_ids, train_attention_masks, train_labels = tokenize_and_encode(train_texts,train_labels)
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(val_texts,val_labels)

print('train_input_ids ',train_input_ids[0].shape ,train_input_ids[0], '\n'
      'train_attention_masks ', train_attention_masks[0] ,train_attention_masks[0], '\n'
      'train_labels', train_labels[0])


class TextClassificationDataset(torch.utils.data.Dataset):
  def __init__(self, input_ids, attention_masks, labels, num_classes=2):
      self.input_ids = input_ids
      self.attention_masks = attention_masks
      self.labels = labels
      self.num_classes = num_classes
      self.one_hot_labels = self.one_hot_encode(labels,num_classes)

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
      return {
          'input_ids':self.input_ids[idx],
          'attention_mask':self.attention_masks[idx],
          'labels':self.one_hot_labels[idx]
      }

  @staticmethod
  def one_hot_encode(targets, num_classes):
    targets = targets.long()
    one_hot_targets = torch.zeros(targets.size(0), num_classes)
    one_hot_targets.scatter_(1, targets.unsqueeze(1), 1.0)
    return one_hot_targets


train_dataset = TextClassificationDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TextClassificationDataset(val_input_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=8)

print(len(train_dataset))
len(val_dataset)

item=next(iter(train_dataloader))
item_ids,item_mask,item_labels=item['input_ids'],item['attention_mask'],item['labels']
print ('item_ids, ',item_ids.shape, '\n',
       'item_mask, ',item_mask.shape, '\n',
       'item_labels, ',item_labels.shape, '\n',)


model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',num_labels=2)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# 모델 및 옵티마이저 준비
accelerator = Accelerator()
model,optimizer,train_dataloader,eval_dataloader = accelerator.prepare(
    model,optimizer,train_dataloader,eval_dataloader
)

# 메트릭 함수 가져오기
from sklearn.metrics import accuracy_score

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))


for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    preds = []
    out_label_ids = []
    epochs = 1
    epoch = 1

    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits

        preds.extend(torch.argmax(logits.detach().cpu(),dim=1).numpy())
        out_label_ids.extend(torch.argmax(inputs['labels'].detach().cpu(),dim=1).numpy())

    accuracy = accuracy_score(out_label_ids, preds)

    print(f"Epoch {epoch + 1}/{num_epochs} Evaluation Results:")
    print(f"Accuracy: {accuracy}")


from transformers import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def inference(text, model,  label, device=device):
    # 토크나이저 불러오기 및 입력 텍스트 토큰화
    inputs = tokenizer(text,return_tensors='pt',padding=True,truncation=True)
    # 입력 텐서를 특정 디바이스로 전송 (디폴트 : cpu)
    inputs = {k:v.to(device) for k,v in inputs.items()}

    # 모델을 위한 eval 모드로 설정 후 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # predicted label 인덱스 추출
    pred_label_idx = torch.argmax(logits.detach().cpu(),dim=1).item()
    print(f'Predicted label index : {pred_label_idx}, actual label {label}')

    return pred_label_idx

text="""
WASHINGTON (ABC) A confirmed tornado was located near Bridgeville in Sussex County, Delaware, shortly after 6 p.m. ET Saturday, moving east at 50 mph, according to the National Weather Service. Downed trees and wires were reported in the area.
"""
inference(text, model, 1.0)
text="this is definately junk text I am typing"
inference(text, model, 0.0)
