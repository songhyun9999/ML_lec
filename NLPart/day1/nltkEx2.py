from nltk.tokenize import sent_tokenize, word_tokenize

sentence = 'Natural language preprocessing (NLP) is a sufield of computer science, information engineering, '\
            'and artificial intelligence cencerned with the interaction between computer and human language. it is a good day'

print(sent_tokenize(sentence))
print(len(sent_tokenize(sentence)))
print(word_tokenize(sentence))
print()

# 품사 분류
from nltk.tag import pos_tag

sentence = 'Smith woodhouse, hansome, clever, and rich, with a comfortable home and happy disposition'
tagged_list = pos_tag(word_tokenize(sentence))
print(tagged_list)
print()


'''
NN	명사(단수)	cat, book
NNS	명사(복수)	cats, books
NNP	고유명사(단수)	John, Google
NNPS	고유명사(복수)	Americans
VB	동사(base)	run, eat
VBD	동사(과거형)	ran, ate
VBG	동사(현재분사/동명사)	running, eating
VBN	동사(과거분사)	eaten, run
VBP	동사(현재형, 복수 주어)	run, eat
VBZ	동사(현재형, 단수 주어)	runs, eats
JJ	형용사	big, red
RB	부사	quickly, very
IN	전치사/접속사	in, of, that
DT	한정사	the, a, some
PRP	인칭대명사	I, you, he
PRP$	소유대명사	my, your, his
CC	등위 접속사	and, but, or
TO	to (to부정사)	to go
MD	조동사	can, will, must
'''

nouns = [w[0] for w in tagged_list if w[1] == 'NN']
print(nouns)


