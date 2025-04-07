import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

raw_text = 'a barber is a person. a barber is good person. as barber is huge person. '\
           'he knew a secret! The secret he kept is huge secret. Huge secret.'\
           'His barber kept his word. a barber kept his word. His barber hept his secret.'\
           'But keeping and keeping such a huge secret to himself was driving the barber crazy. '\
           'the barber went up a huge mountain.'

sentences = sent_tokenize(raw_text)
print(sentences)


vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))
for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    # print(tokenized_sentence)
    result = []

    for word in tokenized_sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)

print(preprocessed_sentences)
print(vocab)
print()

word_to_index = {}
vocab_sorted = sorted(list(vocab.items()),key=lambda x:x[1],reverse=True)
print(vocab_sorted)
i=0

for word,frequency in vocab_sorted:
    if frequency>1:
        i+=1
        word_to_index[word] = i

print(word_to_index)
print()

vocab_size = 5
word_frequency = [word for word,index in word_to_index.items() if index >= vocab_size +1]
for w in word_frequency:
    del word_to_index[w]
print(word_to_index)
print()
word_to_index['OOV'] = len(word_to_index) + 1 # out of vocaburary
print(word_to_index)
print(preprocessed_sentences)

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)
print()

from collections import Counter

print(preprocessed_sentences)
all_word_list = sum(preprocessed_sentences,[])
print(all_word_list)

vocab = Counter(all_word_list)
print(vocab)
print()

vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)

word_to_index = {}
i=0
for word,frequency in vocab:
    if frequency>1:
        i+=1
        word_to_index[word] = i
print(word_to_index)

from nltk import FreqDist

vocab = FreqDist(np.hstack(preprocessed_sentences))
print(vocab.items())
vocab = vocab.most_common(vocab_size)
print(vocab)
word_to_index = {word[0]:index+1 for index, word in enumerate(vocab)}
print(word_to_index)

