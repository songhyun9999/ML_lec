import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import os
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import logging


root_dir = './'
save_dir = os.path.join(root_dir, "save")
output_dir = os.path.join(root_dir, "output")
cache_dir = os.path.join(root_dir, "cache")

config = {"mode": "train",
          "train_data_path": os.path.join(root_dir, "data/train.txt"),
          "test_data_path": os.path.join(root_dir, "data/test.txt"),
          "output_dir_path": output_dir,
          "save_dir_path": save_dir,
          "cache_dir_path": cache_dir,
          "pretrained_model_name_or_path": "hyunwoongko/kobart",
          "max_length": 250,
          "max_dec_length": 60,
          "epoch": 30,
          "batch_size": 16,
          "seed": 1234,
          'device':'cuda' if torch.cuda.is_available() else 'cpu'
          }

logger = logging.getLogger(__name__)

# 재현성을 위한 seed 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# txt 파일 읽어와서 article, title 분리
def read_data(file_path):
    datas = []
    with open(file_path,'r',encoding='utf-8') as f:
        for idx, line in enumerate(f):
            pieces = line.strip().split('\t')
            article, title = pieces[0], pieces[1]
            datas.append((article,title))

    return datas

def convert_data2feature(datas,max_length,max_dec_length,tokenizer):
    input_ids_features, attention_mask_features, decoder_input_features, decoder_attention_mask_features, label_features = [], [], [], [], []

    for article, title in tqdm(datas,desc='convert_data2feature'):
        tokenized_article = tokenizer.tokenize(article)
        tokenized_title = tokenizer.tokenize(title)

        input_ids = tokenizer.convert_tokens_to_ids(tokenized_article)
        attention_mask = [1] * len(input_ids)

        # decoder input start,end symbol 추가
        start_symbol = tokenizer.tokenize('<s>')
        end_symbol = tokenizer.tokenize('</s>')
        decoder_token = start_symbol + tokenized_title + end_symbol

        decoder_input = tokenizer.convert_tokens_to_ids(decoder_token)
        decoder_attention_mask = [1] * len(decoder_input)

        # encoder input 에 padding 추가 -> 고정된 크기의 vector 생성
        pad_id = tokenizer.convert_tokens_to_ids('<pad>')
        padding = [pad_id] * (max_length-len(input_ids))
        input_ids += padding
        attention_mask += padding

        # decoder input 에 padding 추가 -> 고정된 크기의 vector 생성
        dec_padding = [pad_id] * (max_dec_length - len(decoder_input))
        decoder_input += dec_padding
        decoder_attention_mask += dec_padding

        # label 추가
        label = tokenizer.convert_tokens_to_ids(tokenized_title)
        label_padding = [pad_id] * (max_dec_length - len(label))
        label += label_padding

        # 변환한 데이터 리스트에 저장
        input_ids_features.append(input_ids[:max_length])
        attention_mask_features.append(attention_mask[:max_length])
        decoder_input_features.append(decoder_input[:max_dec_length])
        decoder_attention_mask_features.append(decoder_attention_mask[:max_dec_length])
        label_features.append(label[:max_dec_length])

    # 리스트 데이터 텐서 변환
    input_ids_features = torch.tensor(input_ids_features,dtype=torch.long)
    attention_mask_features = torch.tensor(attention_mask_features,dtype=torch.long)
    decoder_input_features = torch.tensor(decoder_input_features,dtype=torch.long)
    decoder_attention_mask_features = torch.tensor(decoder_attention_mask_features,dtype=torch.long)
    label_features = torch.tensor(label_features,dtype=torch.long)

    return input_ids_features, attention_mask_features, decoder_input_features, decoder_attention_mask_features, label_features


def train(model):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config["pretrained_model_name_or_path"])

    """ Train the model """
    # 학습 및 평가 데이터 읽기
    train_datas = read_data(config["train_data_path"])
    test_datas = read_data(config["test_data_path"])

    # 입력 데이터 전처리
    train_input_ids_features, train_attention_mask_features, train_decoder_input_features, train_decoder_attention_mask_features, train_label_features = \
        convert_data2feature(train_datas, config["max_length"], config["max_dec_length"], tokenizer)
    test_input_ids_features, test_attention_mask_features, test_decoder_input_features, test_decoder_attention_mask_features, test_label_features = \
        convert_data2feature(test_datas, config["max_length"], config["max_dec_length"], tokenizer)

    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    train_features = TensorDataset(train_input_ids_features, train_attention_mask_features,
                                   train_decoder_input_features, train_decoder_attention_mask_features,
                                   train_label_features)
    train_dataloader = DataLoader(train_features, sampler=RandomSampler(train_features),
                                  batch_size=config["batch_size"])

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_features = TensorDataset(test_input_ids_features, test_attention_mask_features, test_decoder_input_features,
                                  test_decoder_attention_mask_features, test_label_features)
    test_dataloader = DataLoader(test_features, sampler=SequentialSampler(test_features),
                                 batch_size=config["batch_size"])

    # 모델 학습을 위한 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    set_seed(config["seed"])

    for epoch in range(config["epoch"]):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            model.train()
            batch = tuple(t.cuda() for t in batch)
            outputs = model(input_ids=batch[0],
                            attention_mask=batch[1],
                            decoder_input_ids=batch[2],
                            decoder_attention_mask=batch[3],
                            labels=batch[4],
                            return_dict=True)

            loss = outputs["loss"]

            loss.backward()
            if (global_step + 1) % 50 == 0:
                print("{} Processed.. Total Loss : {}".format(global_step + 1, loss.item()))

            tr_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            # Save model checkpoint
            if global_step % 500 == 0:
                evaluate(model, tokenizer, test_dataloader)

                output_dir = os.path.join(config["output_dir_path"], "checkpoint-{}".format(global_step))
                print("Model Save in {}".format(output_dir))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)

    return global_step, tr_loss / global_step

def evaluate(model, tokenizer, test_dataloader):
    model.eval()
    for batch in tqdm(test_dataloader):
        batch = tuple(t.cuda() for t in batch)

        dec_outputs = model.generate(input_ids = batch[0],
                                     attention_mask=batch[1],
                                     max_length=config["max_dec_length"],
                                     eos_token_id=1,
                                     do_sample=False,
                                     bad_words_ids=[[5]]
                                    )

        batch_size = batch[0].size()[0]

        dec_outputs = dec_outputs.tolist()
        dec_labels = batch[4].tolist()

        for index in range(batch_size):
            if 1 in dec_outputs[index]:
                dec_outputs[index] = dec_outputs[index]
            if -100 in dec_labels[index]:
                dec_labels[index] = dec_labels[index][:dec_labels[index].index(-100)]
            pred = "".join(tokenizer.convert_ids_to_tokens(dec_outputs[index][1:])).replace("Ġ", " ").replace("<pad>", "").replace("</s>", "").replace("▁", " ")
            ref = "".join(tokenizer.convert_ids_to_tokens(dec_labels[index][:-1])).replace("Ġ", " ").replace("<pad>", "").replace("</s>", "").replace("▁", " ")

            print("REFERENCE : {}\nDECODED   : {}\n".format(ref, pred))

if __name__ == '__main__':
    set_seed(config['seed'])
    device = config['device']
    print(f'using {device} for train')

    model = BartForConditionalGeneration.from_pretrained(config["pretrained_model_name_or_path"]).to(config['device'])
    # print(tokenizer)
    train(model)