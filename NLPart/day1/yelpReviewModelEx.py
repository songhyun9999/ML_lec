from argparse import Namespace
from collections import Counter
import json
import os
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import vectorize
from torch.utils.data import Dataset, DataLoader

#DataSet
class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
    ##################################################################
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train':(self.train_df,self.train_size),
                             'val':(self.val_df,self.val_size),
                             'test':(self.test_df,self.test_size)}

        self.set_split('train')

    ##################################################################

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
    ##############################################################################
        review_df = pd.read_csv(review_csv)
        train_review_df = review_df[review_df['split']=='train']
        return cls(review_df,ReviewVectorizer.from_dataframe(train_review_df))
    ##############################################################################
    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):

        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
    ##################################################################
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    ##################################################################

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
    ##################################################################
        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)
        return {'x_data':review_vector,
                'y_target':rating_index}
    ##################################################################


    def get_num_batches(self, batch_size):
        return len(self) // batch_size




#vocabulary
class Vocabulary(object):
    """ 매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스 """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
            add_unk (bool): UNK 토큰을 추가할지 지정하는 플래그
            unk_token (str): Vocabulary에 추가할 UNK 토큰
        """
        #######################################################################################
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx:token for token,idx in self._token_to_idx.items()}

        self.add_unk = add_unk
        self.unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)
        #####################################################################################3

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self.add_unk,
                'unk_token': self.unk_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        ########################################################
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index
        ########################################################

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        ########################################################
        if self.unk_index >=0:
            return self._token_to_idx.get(token,self.unk_index)
        else:
            return self._token_to_idx[token]
        ########################################################

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


#Vectorizer
#리뷰 텍스트를 수치 벡터로 변환하는 클래스
class ReviewVectorizer(object):
    """ 어휘 사전을 생성하고 관리합니다 """

    def __init__(self, review_vocab, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    #one hot encoding
    def vectorize(self, review):
    ##############################################################################3
        vector = [0] * len(self.review_vocab)
        for word in review:
            idx = self.review_vocab.lookup_token(word)
            if idx is not None:
                vector[idx] = 1
        return vector
    ###############################################################################

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
    ###########################################################################
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(' '):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word,count in word_counts.items():
            if count>cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)
    ###########################################################################
    @classmethod
    def from_serializable(cls, contents):
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable()}



#DataLoader
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    #######################################################################################
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle,drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict

    #######################################################################################

#ReviewClassifier 모델
class ReviewClassifier(nn.Module):
    ##################################################################
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_features,out_features=1)

    def forward(self,x_in,apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out

    ##################################################################


#훈련
def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


args = Namespace(
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='data/yelp/reviews_with_splits_list.csv',
    save_dir='model_storage/day1/yelp/',
    vectorizer_file='vectorizer.json',
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)

    print("파일 경로: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

# CUDA 체크

##################################################################
if not torch.cuda.is_available():
    args.cuda = False

print("CUDA 사용여부: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")
set_seed_everywhere(args.seed, args.cuda)

##################################################################



#helper 함수
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    ##################################################################
    if train_state['epoch_index']==0:
        torch.save(model.state_dict(),train_state['model_filename'])
        train_state['stop_early'] = False
    elif train_state['epoch_index'] >= 1:
        loss_t = train_state['val_loss'][-1]
        if loss_t > train_state['early_stopping_best_val']:
            train_state['early_stopping_step'] += 1
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(),train_state['model_filename'])
            train_state['early_stopping_step'] = 0
        train_state['stop_ealry'] = train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state
    ##################################################################


def compute_accuracy(y_pred, y_target):
    ##################################################################
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

    ##################################################################


print()
#초기화
##################################################################
dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

loss_func = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(),lr=args.learning_rate)
##################################################################
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=1)
#learning rate를 감소시킬 비율입니다. 기본값은 0.1로, 검증 손실값이 개선되지 않을 때 현재 learning rate에 0.1을 곱하여 감소시킵니다.
#검증 손실값이 개선되지 않은 상태를 얼마나 허용할 것인지를 설정하는 정수값입니다. 기본값은 10으로,
# 10번의 연속적인 epoch 동안 검증 손실값이 개선되지 않으면 learning rate를 감소시킵니다.
train_state = make_train_state(args)
try:
    for epoch_index in range(args.num_epochs):
        ##################################################################
        train_state['epoch_index'] = epoch_index
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)

        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        print('epoch:{}'.format(epoch_index))

        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()
            y_pred = classifier(x_in=batch_dict['x_data'].float())
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            loss.backward()
            optimizer.step()
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        print('train_loss:{},  train_acc:{}'.format(running_loss, running_acc))

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        dataset.set_split('val')
        batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = classifier(x_in=batch_dict['x_data'].float())
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        print('va_loss:', running_loss)
        print('va_acc:', running_acc)
        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=args,model=classifier,train_state=train_state)
        scheduler.step(train_state['val_loss'][-1])
        if train_state['stop_early']:
            break




        ##################################################################

except KeyboardInterrupt:
    print("Exiting loop")
