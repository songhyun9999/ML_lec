import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
  device="cpu"
  if torch.cuda.is_available():
    device="cuda"
  else:
    device="cpu"
  return device


device = get_device()
print(device)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.block_size = block_size
        self.tokenizer = tokenizer

        with open(file_path, 'r') as f:
            self.data = f.read()

        self.examples = []

        '''
        padding='max_length': 항상 max_length 길이로 맞추기 위해 필요 없는 부분은 [PAD] 토큰으로 채워요.

        truncation=True: 만약 example이 max_length보다 길면 자르도록 해요.
        
        max_length=block_size: 길이를 block_size에 맞춰요.
        
        return_tensors='pt': 결과를 파이토치 텐서로 받아요. (input_ids, attention_mask 등이 텐서 형태로 나옴)
        '''
        for i in range(0, len(self.data)-self.block_size, self.block_size):
            example = self.data[i:i+self.block_size]
            tokenized = self.tokenizer(example, padding='max_length', truncation=True, max_length=block_size, return_tensors='pt')
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]['input_ids'].squeeze()
        attention_mask = self.examples[idx]['attention_mask'].squeeze()
        return input_ids, attention_mask

filename = 'data/input.txt'
train_dataset = ShakespeareDataset(filename, tokenizer)

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    inputs, masks = zip(*batch)
    inputs = torch.stack(inputs).transpose(0, 1)
    masks = torch.stack(masks)
    return inputs, masks
train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn,shuffle=True)

# 데이터의 차원 확인
item=next(iter(train_dataloader))
input_ids,attention_masks=item
print(input_ids.shape, attention_masks.shape)

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
        self.register_buffer('postional_encoding', postional_encoding)
    def forward(self, x):
        x = x + self.postional_encoding[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, dropout):
        super().__init__()

        self.memory_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.memory_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tgt_pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8,
                                       dim_feedforward=2048,
                                       dropout=dropout),
            num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.d_model=embedding_dim
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1

        # 임베딩 층 초기화
        nn.init.uniform_(self.memory_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.tgt_embedding.weight, -initrange, initrange)

        # 디코딩 층 초기화
        for param in self.decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # 출력 층 초기화
        nn.init.uniform_(self.fc.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tgt,  memory=None, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None,tgt_key_padding_mask=None):
        tgt = self.tgt_embedding(tgt) * self.d_model ** 0.5 #scaling
        tgt=self.tgt_pos_encoder(tgt)

        memory=self.memory_embedding(memory) * self.d_model ** 0.5 # memory : encoder output
        memory=self.memory_pos_encoder(memory)

        output = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask,
            memory_mask=memory_mask,#디코더가 인코더의 출력(memory) 중 특정 위치를 보지 못하게 막는 역할.
            memory_key_padding_mask=memory_key_padding_mask,#인코더 입력에서 PAD 토큰에 해당하는 부분을 디코더가 무시하도록 만듬
            tgt_key_padding_mask=tgt_key_padding_mask #디코더 입력(tgt) 중에서 패딩(<PAD>)이 있는 위치를 무시
            )
        print(output)
        output = self.fc(output)
        return output

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)


def create_mask(src, tgt,tokenizer_src=tokenizer,tokenizer_tgt=tokenizer):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == tokenizer_src.pad_token_id).transpose(0, 1)
    tgt_padding_mask = (tgt == tokenizer_tgt.pad_token_id).transpose(0, 1)
    return src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)


model = TransformerDecoder(vocab_size=tokenizer.vocab_size, embedding_dim=768, num_layers=3, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# 데이터셋과 토크나이저 불러오기
dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split="train[:1%]", trust_remote_code=True)  # Only use a 1% portion of the dataset
tokenizer_src = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-german-cased")
print(dataset[0])

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, max_length=50):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx]['translation']['en']
        tgt_text = self.dataset[idx]['translation']['de']

        src_tokens = self.tokenizer_src.encode_plus(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tgt_tokens = self.tokenizer_tgt.encode_plus(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return src_tokens["input_ids"].squeeze(),tgt_tokens["input_ids"].squeeze()

train_data = TranslationDataset(dataset, tokenizer_src, tokenizer_tgt)
# torch.tensor 생성 후 샘플 데이터 확인
print(train_data[2])

def collate_fn(batch):
    src_ids ,tgt_ids = zip(*batch)
    src_ids = torch.stack(src_ids).transpose(0, 1)
    tgt_ids = torch.stack(tgt_ids).transpose(0, 1)
    return src_ids, tgt_ids
dataloader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

item=next(iter(dataloader))
src_ids,tgt_ids=item
print('src_ids ',src_ids.shape)
print(' tgt_ids ',tgt_ids.shape)


