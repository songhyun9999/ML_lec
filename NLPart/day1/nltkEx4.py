import nltk


print(nltk.corpus.gutenberg.fileids())
print()

emma_txt = nltk.corpus.gutenberg.raw('austen-emma.txt')
print(emma_txt)
print()

from nltk.tokenize import RegexpTokenizer

rt = RegexpTokenizer('[\w]+')
print(rt.tokenize(emma_txt[:200]))


