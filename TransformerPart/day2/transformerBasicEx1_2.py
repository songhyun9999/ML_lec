import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
  device="cpu"
  if torch.cuda.is_available():
    device="cuda"
  else:
    device="cpu"
  return device


device = get_device()
print(device)

##################################################################################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding, dropout=0.1, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        postional_encoding = torch.zeros(max_seq_len, dim_embedding)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denom_term = torch.exp(torch.arange(0, dim_embedding, 2).float() * (-math.log(10000.0) / dim_embedding))
        postional_encoding[:, 0::2] = torch.sin(position * denom_term)
        postional_encoding[:, 1::2] = torch.cos(position * denom_term)
        postional_encoding = postional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('postional_encoding', postional_encoding)#positional encoding은 학습할 필요는 없지만, 모델 저장이나 GPU 이동할 때 반드시 같이 가야 하니까 PyTorch에게 특별히 알려주는 것."
    def forward(self, x):
        x = x + self.postional_encoding[:x.size(0), :]
        return self.dropout(x)


from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch

# 데이터셋과 토크나이저 불러오기
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 데이터셋 토큰화
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=512)

train_dataset = dataset["train"].map(tokenize, batched=True, batch_size=len(dataset["train"]))
val_dataset = dataset["test"].map(tokenize, batched=True, batch_size=len(dataset["test"]))

# 토큰화된 데이터셋에서 input_ids와 attention_mask 추출
train_data = torch.tensor(train_dataset["input_ids"])
train_attention_mask = torch.tensor(train_dataset["attention_mask"])
train_labels = torch.tensor(train_dataset["label"])

val_data = torch.tensor(val_dataset["input_ids"])
val_attention_mask = torch.tensor(val_dataset["attention_mask"])
val_labels = torch.tensor(val_dataset["label"])

# TensorDataset 생성
train_dataset = TensorDataset(train_data, train_attention_mask, train_labels)
val_dataset = TensorDataset(val_data, val_attention_mask, val_labels)

# DataLoader 생성
def collate_fn(batch):
    input_ids, attention_mask, labels = zip(*batch)
    input_ids = torch.stack(input_ids).transpose(0, 1) # input_ids 트랜스포즈(Transpose)#huggingface(transformer) 모델은 seq, batch
    attention_mask = torch.stack(attention_mask)       # attention_mask 트랜스포즈(Transpose)
    labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=2).float().to(device)
    return input_ids, attention_mask, labels


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class TextClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, nhead, num_layers, num_classes):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        # 트랜스포머 인코더 층 생성
        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim=embedding_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, key_padding_mask=None):
        '''
        x는 입력된 토큰 인덱스를 임베딩 벡터로 변환한 후, 임베딩 차원의 제곱근을 곱해 줍니다. 이는 임베딩 크기에 의한 값의 스케일 차이를 보정하는 역할을 합니다.
        '''
        x = self.embedding(x)* math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)#src_key_padding_mask는 패딩 마스크로, 시퀀스에 포함된 패딩 토큰을 무시할 수 있게 도와줍니다.

        # 첫 번째 차원을 기준으로 나머지 차원(마지막 차원) 값의 평균값 생성
        '''
        트랜스포머 인코더의 출력은 시퀀스 차원에 대한 정보가 담겨 있습니다. mean(dim=0)을 사용하여 시퀀스 차원에 대해 평균을 구합니다. 
        이렇게 하면 시퀀스의 전체적인 의미를 추출할 수 있습니다.
        '''
        x = x.mean(dim=0)

        # 분류 작업용 완전 연결 층
        x = self.fc(x)
        x=torch.sigmoid(x)
        return x

import torch.optim as optim  # optim 모듈 임포트 추가
import torch.nn as nn

vocab_size = tokenizer.vocab_size
embedding_dim = 512
nhead = 8
num_layers = 6
num_classes = 2

# 모델 생성
model = TextClassifier(vocab_size, embedding_dim, nhead, num_layers,  num_classes).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10
# for epoch in range(num_epochs):
#     i=0
#     for batch_data, batch_attention_mask, batch_labels in train_dataloader:
#
#         optimizer.zero_grad()
#
#         # attention_mask를 불리언(boolean) 텐서로 변환
#         batch_attention_mask = (batch_attention_mask==0).to(device)
#
#         outputs = model(batch_data.to(device), key_padding_mask=batch_attention_mask)
#         loss = criterion(outputs, batch_labels.to(device))
#         if i%100==0:
#           print ("epoch ", epoch, "batch ", i, "loss ", loss)
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
#         i=i+1
#
#     print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
#
#
# torch.save(model.state_dict(), "model_save/TextClassificationModel.pth")
# print()

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
embedding_dim = 512
nhead = 8
num_layers = 6
num_classes = 2

# 모델 생성
model_loaded = TextClassifier(vocab_size, embedding_dim, nhead, num_layers,  num_classes).to(device)

# 학습 모델 가중치(weights) 불러오기
model_loaded.load_state_dict(torch.load('model_save/TextClassificationModel.pth'))
model_loaded.eval()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params)

# 주어진 텍스트에서 추론을 실행하는 함수
def infer(text):
    # 입력 텍스트 토큰화
    tokens = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = tokens["input_ids"].to(device).transpose(0,1)

    attention_mask = tokens["attention_mask"]
    attention_mask=(attention_mask==0).to(device)
    print(input_ids.shape)
    print(attention_mask)

    # 추론 실행
    with torch.no_grad():
        output = model_loaded(input_ids, key_padding_mask=attention_mask)
    # 출력을 클래스 확률로 변환
    probabilities = output.squeeze(0)
    return probabilities


# 샘플 텍스트로 테스트
example_text = "This movie is good! ."
probabilities = infer(example_text)

print("Probabilities:", probabilities)
