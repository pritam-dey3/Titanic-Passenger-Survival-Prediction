from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import joblib

Pclass_ix = np.s_[:, 0]
SibSp_ix = np.s_[:, 1]
Parch_ix = np.s_[:, 2]
Sex_ix = np.s_[:, 3:5]
Age_ix = np.s_[:, 5]
Fare_ix = np.s_[:, 6]
Embarked_ix = np.s_[:, 7:10]


class ImputeFare(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = XGBClassifier(max_depth=3, max_leaf_nodes=3, n_estimators=97)
    def fit(self, X=None, y=None):
        y = X[Fare_ix]
        nan_ix = np.isnan(y.astype(float))
        X = X[~nan_ix]
        y = y[~nan_ix]
        X = np.c_[X[Pclass_ix], X[Sex_ix],
            X[Embarked_ix], X[SibSp_ix] + X[Parch_ix]]
        self.model = self.model.fit(X, y)
        return self
    def transform(self, X):
        y = X[Fare_ix]
        nan_ix = np.isnan(y.astype(float))
        #select predictor set
        X_pred = np.c_[X[Pclass_ix], X[Sex_ix],
            X[Embarked_ix], X[SibSp_ix] + X[Parch_ix]]
        X_pred = X_pred[nan_ix]
        #select features
        #replace nan variables
        X[nan_ix, Fare_ix[1]] = self.model.predict(X_pred)
        return X 

class ImputeAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = XGBClassifier(eta=0.14, n_estimators=26, reg_lambda=0.9)
    def fit(self, X=None, y=None):
        y = X[Age_ix]
        nan_ix = np.isnan(y.astype(float))
        X = X[~nan_ix]
        y = y[~nan_ix]
        X = np.c_[X[Pclass_ix], X[Parch_ix], X[SibSp_ix], 
                X[Sex_ix], X[Fare_ix]]
        self.model = self.model.fit(X, y)
        return self
    def transform(self, X):
        y = X[Age_ix]
        nan_ix = np.isnan(y.astype(float))
        #select predictor set
        X_pred = np.c_[X[Pclass_ix], X[Parch_ix], X[SibSp_ix], 
                X[Sex_ix], X[Fare_ix]]
        X_pred = X_pred[nan_ix]
        #replace nan variables
        X[nan_ix, Age_ix[1]] = self.model.predict(X_pred)
        return X

class DiscreteVar(BaseEstimator, TransformerMixin):
    def __init__(self, bins=[-1, 5, 10, 35, 100, 250, np.inf], features=[]):
        self.bins = bins
    def fit(self, X=None, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.DataFrame(X).iloc[:, 0] 
        X = pd.cut(X, bins=self.bins, labels=np.arange(len(self.bins[:-1])))
        # print(np.c_[X.to_numpy()].shape)
        return np.c_[X.to_numpy()]