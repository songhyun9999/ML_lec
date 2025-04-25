import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#
# # 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

#
# # 데이터셋 불러오기
dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
'''
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 1
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 1
    })
    test: Dataset({
        features: ['text'],
        num_rows: 1
    })
})
'''

# 연속형 텍스트를 작은 청크로 분리
def split_text(text,max_length=100):
    return [text[i:i+max_length] for i in range(0,len(text),max_length)]

# split_text 함수를 데이터셋에 적용
split_texts = split_text(dataset["train"]["text"][0])
print(split_texts)

# split_texts 변수에 담긴 값을 토큰화
tokenized_texts = tokenizer(split_texts, return_tensors='pt',
                            padding=True,truncation=True)


class ShiftedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        labels = input_ids[1:].tolist() + [tokenizer.eos_token_id]

        return {'input_ids':input_ids,'attention_mask':attention_mask,'labels':torch.tensor(labels)}

    def __len__(self):
        return len(self.encodings["input_ids"])

# # DataLoader 생성
train_dataset = ShiftedDataset(tokenized_texts)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
item=next(iter(train_dataloader))
print(item['input_ids'])
print(item['attention_mask'])
print(item['labels'])


from accelerate import Accelerator
from transformers import GPT2LMHeadModel

# Accelerator 초기화
accelerator = Accelerator()

# training arguments 설정
num_epochs = 10
learning_rate = 5e-5

# GPT-2 모델 및 옵티마이저 초기화
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Accelerator와 함께 학습시키기 위해 모델과 옵티마이저 준비
model, optimizer, train_dataloader = accelerator.prepare(model,
                                                         optimizer,
                                                         train_dataloader)

epoch=10

from torch.optim import AdamW
from tqdm import tqdm

# 파인튜닝 반복 루프
for epoch in range(num_epochs):
    epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    for step, batch in enumerate(epoch_iterator):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        if step % 500 == 0:
            epoch_iterator.set_postfix({"Loss": loss.item()}, refresh=True)

    # 매 5회 에포크마다 모델 저장
    # 여러분 구글 드라이브의 알맞은 경로로 model_save_path 지정
    if (epoch + 1) % 5 == 0:
        model_save_path = f"model_save/text_generator_model/model_checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(model_save_path)
        print(f"Model saved at epoch {epoch + 1}")

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)


model_path = 'model_save/text_generator_model/model_checkpoint_epoch_10'
tokenizer_path = 'gpt2'

unwrapped_model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)


from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

def generate_poem(prompt, model_path, tokenizer_path, max_words=50, max_seq_len=100, temperature=1.0):
    # 파인튜닝 모델 및 토크나이저 불러오기
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    # padding token 및 padding side 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    poem = ''
    remaining_words = max_words

    while remaining_words > 0:
        # 프롬프트 설정 및 text 생성
        input_ids = tokenizer.encode(prompt,return_tensors='pt',padding=True,truncation=True,max_length=max_seq_len)
        attention_mask = torch.ones_like(input_ids)

        max_tokens = min(remaining_words * 5, max_seq_len) # 각 단어가 평균 5개 토큰으로 구성됐다고 가정
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            num_return_sequences=1, # 출력으로 리턴되는 단어수
            no_repeat_ngram_size=2, # size로 구성된 문장이 반복되지 않도록 함
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature, # 높을수록 창작, 낮을수록 정확하게(학습데이터와 가깝게)
        )

        # token IDs 를 text 로 변환
        generated_text = tokenizer.decode(output_ids[0],skip_special_tokens=True)
        poem += generated_text
        remaining_words -= len(generated_text.split())

        # 생성된 text의 마지막 부분으로 프롬프트를 업데이트
        prompt = generated_text.split()[-max_seq_len:]

    return poem

import re

def post_process_poem(poem):
    # 여분의 스페이스(공백) 제거
    poem = re.sub(r'\s+',' ',poem).strip()

    # 각 sentence의 첫 글자를 대문자로 변경
    sentences = re.split(r'(?<=[\.\?!])\s',poem)
    formatted_sentences = [sentence.capitalize()  for sentence in sentences]
    formatted_poem = ' '.join(formatted_sentences)

    # 가독성을 위해 줄 변경(line breaks) 조직
    line_breaks = re.compile(r'(?<=[.;:?!])\s')
    formatted_poem = line_breaks.sub('\n',formatted_poem)

    return formatted_poem


tokenizer_path = 'gpt2'
prompt = "love"
model_path = 'model_save/text_generator_model/model_checkpoint_epoch_10'
max_words = 50
temperature = 0.9  # 이 수치는 randomness 확장 혹은 축소를 위해 조정 가능
generated_poem = generate_poem(prompt, model_path, tokenizer_path, max_words=max_words, temperature=temperature)
formatted_poem = post_process_poem(generated_poem)
print(formatted_poem)


