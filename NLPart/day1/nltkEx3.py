from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

print(stopwords.words('english'))
print(len(stopwords.words('english')))
print()

sentence = 'Family is not an important thing it is everything'
word_tokens = word_tokenize(sentence)
stopwords = set(stopwords.words('english'))
result = []
print(word_tokens)
for w in word_tokens:
    if w not in stopwords:
        result.append(w)

print(result)