import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv')
housing.info()
print()

print(housing['ocean_proximity'].unique())
print(housing['ocean_proximity'].value_counts())
print()

print(housing.describe())
print()
# housing.hist(bins=50, figsize=(20,15))
# plt.show()
import numpy as np
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing.head())
print()
print(housing['income_cat'].value_counts())
# housing['income_cat'].hist()
# plt.show()
print()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]

# strat_train_sets = []
# strat_test_sets = []
#
# for idx, (train_idx, test_idx) in enumerate(split.split(housing, housing['income_cat'])):
#     strat_train_sets[idx] = housing.loc[train_idx]
#     strat_test_sets[idx] = housing.loc[test_idx]

print('original data\n', housing['income_cat'].value_counts()/ len(housing))
print()
print('test data\n', strat_test_set['income_cat'].value_counts()/ len(strat_test_set))

print()
print('train data\n', strat_train_set['income_cat'].value_counts()/ len(strat_train_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = strat_train_set.copy()
housing.info()
print()

pd.set_option('display.max_columns', 20)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
sample_incomplet_row = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplet_row)

print(housing.dropna(subset=['total_bedrooms'])) #option1
print()
print(housing.drop('total_bedrooms', axis=1)) #option2
print()
median = housing['total_bedrooms'].median()
print(sample_incomplet_row['total_bedrooms'].fillna(median)) #option3
print()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
print()

x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x,
                          columns=housing_num.columns,
                          index=housing_num.index)
housing_tr.info()
print()

housing_cat = housing[['ocean_proximity']]

from sklearn.preprocessing import OrdinalEncoder
oridinal_encoder = OrdinalEncoder()
housing_cat_encoded = oridinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(oridinal_encoder.categories_)
print()

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_onehot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_onehot)
print(housing_cat_onehot.toarray())

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)
print()

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
print(num_attribs)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared[0, :])

