import re
import numpy as np
import pandas as pd
import collections
from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv = 'data/yelp/raw_train.csv',
    raw_test_dataset_csv = 'data/yelp/raw_test.csv',
    proportion_subset_of_train = 0.1,
    train_propotion = 0.7,
    val_propotion = 0.15,
    test_propotion = 0.15,
    output_munged_csv = 'data/yelp/reviews_with_splits_list2.csv',
    seed=1337
)

train_reviews = pd.read_csv(args.raw_train_dataset_csv,header=None,names=['rating','review'])

# train_reviews.info()
# print(train_reviews.head()) # 1:negative 2:positive

by_rating = collections.defaultdict(list)

for _,row in train_reviews.iterrows():
    # by_rating[row.rating].append(row.review)
    by_rating[row.rating].append(row.to_dict())

# print(by_rating)

review_subset = []
for _,item_list in sorted(by_rating.items()):
    n_total =len(item_list)
    n_subset = int(args.proportion_subset_of_train * n_total)
    review_subset.extend(item_list[:n_subset])

# print(review_subset[:10])

review_subset = pd.DataFrame(review_subset)
print(review_subset.head())
print(review_subset.tail())
print(review_subset.rating.value_counts())

by_rating = collections.defaultdict(list)

for _,row in review_subset.iterrows():
    # by_rating[row.rating].append(row.review)
    by_rating[row.rating].append(row.to_dict())

final_list = []
np.random.seed(args.seed)

for _,item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(args.train_propotion * n_total)
    n_val = int(args.val_propotion * n_total)
    n_test = int(args.test_propotion * n_total)

    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'

    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'

    final_list.extend(item_list)

final_review = pd.DataFrame(final_list)
print(final_review.head())
print(final_review.tail())
print(final_review.split.value_counts())
print()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'([.,!?])',r' \1 ',text)
    text = re.sub(r'[^a-zA-Z.,!?]',r' ',text)
    return text

final_review.review = final_review.review.apply(preprocess_text)
final_review['rating'] = final_review.rating.apply({1:'negative',2:'positive'}.get)
print(final_review.head())
print(final_review.tail())
print()
print(final_review.iloc[0]['review'])
final_review.to_csv(args.output_munged_csv,index=False)





