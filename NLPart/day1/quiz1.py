from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer

with open('trumph.txt','r',encoding='utf-8') as f:
    data = f.read()

tokenize = RegexpTokenizer("[\w']+")
tdata = tokenize.tokenize(data)

print(tdata)
print()

stopwords = set(stopwords.words('english'))
# print(stopwords)
result = []
for word in tdata:
    if word not in stopwords:
        result.append(word)

from collections import Counter



