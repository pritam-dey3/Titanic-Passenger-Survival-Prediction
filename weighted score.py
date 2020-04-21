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
from functon_scripts.SpecialImputers import DiscreteVar, ImputeAge, ImputeFare
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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

clean_and_impute = Pipeline([
    ('clean', clean_1),
    ('fimp', ImputeFare()),
    ('agimp', ImputeAge())
])


# %%
clean_data = clean_1.fit_transform(data)

Pclass_ix = np.s_[:, 0]
SibSp_ix = np.s_[:, 1]
Parch_ix = np.s_[:, 2]
Sex_ix = np.s_[:, 3:5]
female_ix = np.s_[:, 3]
male_ix = np.s_[:, 4]
Age_ix = np.s_[:, 5]
Fare_ix = np.s_[:, 6]
Embarked_ix = np.s_[:, 7:10]


# %%
def weight_rule(X, y):
    gender_survivor_group = pd.Series(3 * X[female_ix] + X[male_ix] + y)
    weights = [0.1 if x in (1, 4) else 0.4 for x in gender_survivor_group]
    return np.array(weights)


# %%
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import f1_score

class WeightedScorer(_BaseScorer):
    def __init__(self, *args, **kwargs):
        self.get_weights = kwargs.pop('rule')
        super().__init__(*args, **kwargs)
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_pred = method_caller(estimator, "predict", X)
        sample_weight = self.get_weights(X, y_true)
        return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)


# %%
weighted_f1 = WeightedScorer(f1_score, rule=weight_rule, sign=1, kwargs={})


# %%
labels = d1.Survived
data = d1.copy()
data.drop('Survived', axis=1)

clean_data = clean_and_impute.fit_transform(data)


# %%
from sklearn.base import BaseEstimator, ClassifierMixin

class ReturnSex(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self
    def predict(self, X):
        return X[female_ix].astype(int)

clf = ReturnSex()
clf.fit(clean_data, labels)

y_hat = clf.predict(clean_data)


# %%
Weighted_f1(clf, clean_data, labels)


# %%
f1_score(y_hat, labels)


# %%


