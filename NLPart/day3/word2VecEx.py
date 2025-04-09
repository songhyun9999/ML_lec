import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from NLPart.day1.nltkEx5 import word_to_index

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is ploand capital',
    'berlin is germany capital',
    'paris is france capital'
]

def tokenize_corpus(corpus):
    token = [x.split() for x in corpus]
    return token

tokenized_corpus = tokenize_corpus(corpus)
print(tokenized_corpus)

vocaburary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocaburary:
            vocaburary.append(token)

print(vocaburary)

word2idx = {w : idx for idx,w in enumerate(vocaburary)}
idx2word = {idx : w for idx,w in enumerate(vocaburary)}
print(word2idx)
print(idx2word)

window_size = 2
idx_pairs = []

for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # print(indices)
    for center_word_pos in range(len(indices)):
        for w in range(-window_size,window_size+1):
            context_word_pos = center_word_pos + w
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append([indices[center_word_pos],context_word_idx])

idx_pairs = np.array(idx_pairs)
print(idx_pairs)

vocaburary_size = len(vocaburary)
print(vocaburary_size)
print()

def get_input_layer(word_idx):
    x = torch.zeros(vocaburary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5

W1 = torch.tensor(torch.randn(vocaburary_size,embedding_dims),requires_grad=True)
W2 = torch.tensor(torch.randn(embedding_dims,vocaburary_size),requires_grad=True)
total_epochs = 100
learning_rate = 0.001

for epoch in range(total_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x_data = torch.tensor(get_input_layer(data))
        y_target = torch.tensor(torch.from_numpy(np.array([target]))).long()

        z1 = torch.matmul(x_data,W1)
        z2 = torch.matmul(z1,W2)

        log_softmax = F.log_softmax(z2,dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1),y_target)
        loss_val += loss.item()
        loss.backward()

        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if (epoch+1) % 10 == 0:
        print(f'epoch:{epoch+1}, loss:{loss_val/len(idx_pairs):.4f}')

print()

word1 = 'he'
word2 = 'king'
word1vector = torch.matmul(get_input_layer(word2idx[word1]),W1)
print(word1vector)
word2vector = torch.matmul(get_input_layer(word2idx[word2]),W1)
print(word2vector)
