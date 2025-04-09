from argparse import Namespace
import nltk.data
import pandas as pd
import re


args = Namespace(
    raw_dataset_txt = 'data/books/frankenstein.txt',
    window_size = 5,
    train_proportion = 0.7,
    val_proportion = 0.15,
    test_proportion = 0.15,
    output_munged_csv = 'data/books/frankenstein_with_split2.csv',
    seed = 1337
)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open(args.raw_dataset_txt) as fp:
    book = fp.read()

sentences = tokenizer.tokenize(book)
# print(sentences)
# print(len(sentences))
# print(sentences[100])
# print()

def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split())
    text = re.sub(r'([.,!?])',r' \1 ',text)
    text = re.sub(r'[^a-zA-Z.,!?]+',r' ',text)
    return text

cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
print(cleaned_sentences[0])
print()

MASK_TOKEN = '<MASK>'
flatten = lambda outer_list:[item for inner_list in outer_list for item in inner_list]
windows = flatten(list(nltk.ngrams([MASK_TOKEN] * args.window_size + sentence.split() + [MASK_TOKEN]*args.window_size,
                 args.window_size*2+1) for sentence in cleaned_sentences))
print(windows[:5])
print(len(windows))

data = []
for window in windows:
    target_token = window[args.window_size]
    context = []
    for i,token in enumerate(window):
        if token == MASK_TOKEN or i == args.window_size:
            continue
        else:
            context.append(token)
    data.append([' '.join(token for token in context),target_token])

cbow_data = pd.DataFrame(data,columns=['context','target'])
print(cbow_data.head())
n = len(cbow_data)
print(n)
print()

def get_split(row_num):
    if row_num < n * args.train_proportion:
        return 'train'
    elif (row_num > n * args.train_proportion) and (row_num<=n*args.train_proportion + n * args.val_proportion):
        return 'val'
    else:
        return 'test'

cbow_data['split'] = cbow_data.apply(lambda row:get_split(row.name),axis=1)
print(cbow_data.head())
print(cbow_data.tail())
cbow_data.to_csv(args.output_munged_csv,index=False)
