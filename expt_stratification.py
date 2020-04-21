# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

d1 = pd.read_csv('data/train.csv')
testdata = pd.read_csv('data/test.csv')


# %%
from sklearn.compose import ColumnTransformer
from functon_scripts.CustomEstimators import DropColumns, EmbarkedImputer
from functon_scripts.SpecialImputers import DiscreteVar
from sklearn.preprocessing import OneHotEncoder

labels = d1.Survived
data = d1.copy()
data = data.drop('Survived', axis=1)

cols = data.columns.drop(['Embarked', 'Age', 'Sex', 'Fare'])
fare_bins = [-1, 5, 10, 35, 100, np.inf]
age_bins = [0, 18, 32, np.inf] # more on this later
clean_1 = ColumnTransformer([
    ('drp', DropColumns(drop_ix=[0, 2, 5, 6]), cols),
    ('ohe', OneHotEncoder(), ['Sex']),
    ('dAge', DiscreteVar(bins=age_bins), ['Age']),
    ('dFare', DiscreteVar(bins=fare_bins), ['Fare']),
    ('EmbImp', EmbarkedImputer(), ['Embarked'])
])


# %%
clean_data = clean_1.fit_transform(data)

Pclass_ix = np.s_[:, 0]
SibSp_ix = np.s_[:, 1]
Parch_ix = np.s_[:, 2]
Sex_ix = np.s_[:, 3:5]
Age_ix = np.s_[:, 5]
Fare_ix = np.s_[:, 6]
Embarked_ix = np.s_[:, 7:10]


# %%
gender_survivor_group = pd.Series(3 * clean_data[:, 3] + clean_data[:, 4] + labels)


# %%
from sklearn.model_selection import StratifiedShuffleSplit

stt = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_ix, val_ix = next(stt.split(data, labels, groups=gender_survivor_group))


# %%
gender_survivor_group.iloc[train_ix].value_counts().sort_index() / 801


# %%
gender_survivor_group.iloc[val_ix].value_counts().sort_index() / 90


# %%



# %%


