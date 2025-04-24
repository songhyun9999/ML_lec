import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


device = get_device()
print(device)

with open('tokenizer_train.txt','r') as f:
    dataset = [line.strip() for line in f.readlines()]
print(dataset)

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE()) # binary pair encoding
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"])
tokenizer.train_from_iterator(dataset,trainer=trainer)

tokenizer.save('tokenizer.json')


from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json"
)

text = 'The tokenizer words'
encoded = tokenizer.encode(text)
print(encoded.tokens)
print(encoded.ids)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize('The tokenize word'))


from datasets import load_dataset
from transformers import AutoTokenizer

imdb_dataset = load_dataset('imdb')
print(imdb_dataset)
print(imdb_dataset['test'])
sample_text = imdb_dataset['test'][0]['text']
print()
print(sample_text)


model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoModelForSequenceClassification, pipeline

model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analysis_pipeline = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)

result = sentiment_analysis_pipeline(sample_text)
print('sample text:',sample_text)
print('sentiment analysis result:',result)

from torch.utils.data import Dataset
import torch

def preprocess(data):
    dataset = []
    for example in data:
        text = example['text'].lower()
        label = example['label']
        dataset.append({'text':text,'label':label})

    return dataset

train_data = preprocess(imdb_dataset['train'])

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = CustomDataset(train_data)
print(train_dataset[0])

