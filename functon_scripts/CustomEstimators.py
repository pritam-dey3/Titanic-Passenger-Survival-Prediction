from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import pandas as pd
# Cabin and Age imputer Estimators

class CabinImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imp='Z'):
        self.imp = imp
        pat = re.compile(r'^[A-Z]')
        self.categorizer = np.vectorize(lambda x: pat.match(x).group())
    def fit(self, X=None, y=None):
        self.imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=self.imp)
        return self 
    def transform(self, X):
        X = self.imputer.fit_transform(X)
        X[:, 0] = self.categorizer(X[:, 0])
        return X
    def get_feature_names(self):
        return np.array(['Cabin'])
    def get_params(self, deep=True):
        return {'imp': self.imp}

class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imp = SimpleImputer(strategy='most_frequent')
        self.ohe = OneHotEncoder()
    def fit(self, X=None, y=None):
        return self
    def transform(self, X):
        X = self.imp.fit_transform(X)
        return self.ohe.fit_transform(X)
    def get_feature_names(self):
        return np.array(['Embarked'])


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, drop_ix=[0, 2, 7], features=[]):
        self.drop_list = drop_ix
        self.features = features
    def fit(self, X=None, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.delete(X, self.drop_list, axis=1)
    def get_params(self, deep=True):
        return {'drop_ix': self.drop_list, 'features': self.features}
    def get_feature_names(self, deep=True):
        return [''] * len(self.drop_list)

class DoNothing(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Age', 'SibSp', 'Parch']):
        self.features = features
    def fit(self, X=None, y=None):
        self.feature_len = X.shape[1]
        return self
    def transform(self, X):
        return X
    def get_params(self, deep=True):
        return {'features': self.features}
    def get_feature_names(self):
        return self.features

class FareCat(BaseEstimator, TransformerMixin):
    """ This assumes that no value is missing in Fare column """
    def __init__(self):
        self.imputing_model = None
    def fit(self, X=None, y=None):
        return self
    def transform(self, X):
        X = pd.Series(X[:, 0])
        X = pd.cut(X, [-1, 0.1, 10, 30, 100, 200, np.inf], labels=np.arange(0, 6))
        return np.reshape(X.to_numpy(), (-1, 1))
    def get_feature_names(self, deep=True):
        return ['Fare']
