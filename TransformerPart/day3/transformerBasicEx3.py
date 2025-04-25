import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tokenizer 설정 (특수 토큰 추가)
tokenizer_src = AutoTokenizer.from_pretrained('bert-base-uncased') # 영어 특화
tokenizer_tgt = AutoTokenizer.from_pretrained('bert-base-german-cased') # 독일어 특화

tokenizer_src.add_special_tokens({'pad_token':'[PAD]'})
tokenizer_tgt.add_special_tokens({'pad_token':'[PAD]'})


#International Workshop on Spoken Language Translation
#IWSLT 2017은 독일어(Deutsch) → 영어(English) 번역을 위한 데이터셋
# 데이터셋 불러오기 (1% 샘플)
dataset = load_dataset('iwslt2017','iwslt2017-de-en',split='train[:1%]',trust_remote_code=True)


# 커스텀 Dataset
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

        src = self.tokenizer_src.encode(src_text,padding='max_length',truncation=True,max_length=self.max_length)
        tgt = self.tokenizer_tgt.encode(tgt_text,padding='max_length',truncation=True,max_length=self.max_length)

        return torch.tensor(src), torch.tensor(tgt)


train_data = TranslationDataset(dataset,tokenizer_src,tokenizer_tgt)

# Collate 함수
def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.stack(src).transpose(0,1) # (seq_len,batch)
    tgt = torch.stack(tgt).transpose(0,1) # (seq_len,batch)

    return src,tgt

dataloader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

# PositionalEncoding 정의
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # model dimension
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model,dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.fc = nn.Linear(d_model,tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(
            src,tgt,src_mask=src_mask,tgt_mask=tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )

        return self.fc(output)

# 마스크 생성 함수
def generate_square_subsequent_mask(sz):
    '''
    sz x sz 크기의 행렬 생성
    상삼각 행렬의 윗부분을 -inf로 채움 (triu)
    나머지는 0
    이 마스크는 현재 시점 이후의 토큰을 가려서 모델이 미래 정보를 보지 못하게 만듦
    예시 (sz=3):
    [[0,   -inf, -inf],
     [0,    0,   -inf],
     [0,    0,     0 ]]

    '''

    return torch.triu(torch.ones(sz,sz)*float('-inf'),diagonal=1).to(device)

def create_mask(src, tgt, pad_id_src, pad_id_tgt):
    src_seq_len, tgt_seq_len = src.size(0), tgt.size(0)
    src_mask = torch.zeros((src_seq_len,src_seq_len),device = device).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_padding_mask = (src == pad_id_src).transpose(0,1)
    tgt_padding_mask = (tgt == pad_id_tgt).transpose(0,1)

    '''
    src = tensor([
    [2, 3, 4, 1],   # 문장 1 (1은 pad 토큰)
    [5, 6, 1, 1]    # 문장 2
    ])  # shape: (batch_size=2, seq_len=4)
    src == pad_id_src
    
    tensor([
    [False, False, False,  True],
    [False, False,  True,  True]
    ])
    '''
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# 모델 생성
model = TransformerModel(
    src_vocab_size = len(tokenizer_src),
    tgt_vocab_size = len(tokenizer_tgt),
    d_model=512
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)

# 학습 루프
num_epochs = 500
model.train()
for epoch in range(num_epochs):
    for i, (src, tgt) in enumerate(dataloader):
        src,tgt = src.to(device),tgt.to(device)

        tgt_input = tgt[:-1,:]
        tgt_out = tgt[1:,:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src,tgt_input,tokenizer_src.pad_token_id,tokenizer_tgt.pad_token_id
        )

        preds = model(src,tgt_input,src_mask,tgt_mask,src_padding_mask,tgt_padding_mask)

        optimizer.zero_grad()
        loss = criterion(preds.reshape(-1,preds.shape[-1]),tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

#  모델 저장하기
torch.save(model.state_dict(), "model_save/transformer_translation.pth")

#  저장한 모델 불러오기
model_loaded = TransformerModel(
    src_vocab_size=len(tokenizer_src),
    tgt_vocab_size=len(tokenizer_tgt),
    d_model=512
).to(device)
model_loaded.load_state_dict(torch.load("model_save/transformer_translation.pth"))
model_loaded.eval()

def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, max_len=50):
    model.eval()
    src = tokenizer_src.encode(sentence,return_tensors='pt',truncation=True,max_length=max_len,
                               padding='max_length').to(device)
    src = src.transpose(0,1) # (seq_len, 1)

    src_mask = torch.zeros((src.size(0),src.size(0)),device=device).type(torch.bool)
    src_padding_mask = (src==tokenizer_src.pad_token_id).transpose(0,1)

    memory = model.transformer.encoder(
        model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model)),
        mask = src_mask,
        src_key_padding_mask = src_padding_mask
    )

    ys = torch.ones(1,1).fill_(tokenizer_tgt.cls_token_id if tokenizer_tgt.cls_token_id else tokenizer_tgt.bos_token_id).type(torch.long).to(device)

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0))
        out = model.transformer.decoder(
            model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model)),
            memory,
            tgt_mask = tgt_mask.to(device),
            memory_key_padding_mask = src_padding_mask
        )
        out = model.fc(out)
        prob = out[-1, 0]
        next_token = prob.argmax().item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_token)], dim=0)
        # tokenizer_tgt.sep_token_id:bert, tokenizer_tgt.eos_token_id:gpt => 둘 다 종료 문자
        if next_token == tokenizer_tgt.sep_token_id or next_token == tokenizer_tgt.eos_token_id:
            break

    tgt_tokens = ys.squeeze().tolist()
    # skip_special_tokens=True => [CLS], [SEP], [PAD] ,[EOS]와 같은 SPECIAL TOKEN을 출력하지 않음
    translation = tokenizer_tgt.decode(tgt_tokens, skip_special_tokens=True)
    return translation


test_sentence = "I love machine learning and natural language processing."
translated = translate_sentence(model_loaded, test_sentence, tokenizer_src, tokenizer_tgt)
print(" 번역 결과:", translated)

