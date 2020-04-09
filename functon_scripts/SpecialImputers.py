from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

Pclass_ix = np.s_[:, 0:3]
Sex_ix = np.s_[:, 3:5]
Cabin_ix = np.s_[:, 5:14]
Embarked_ix = np.s_[:, 14:17]
Age_ix = np.s_[:, 17]
SibSp_ix = np.s_[:, 18]
Parch_ix = np.s_[:, 19]
Fare_ix = np.s_[:, 20]

class ImputeFare(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = XGBClassifier(max_depth=3, max_leaf_nodes=3, n_estimators=30)
    def fit(self, X=None, y=None):
        y = X[Fare_ix]
        nan_ix = np.isnan(y.astype(float))
        X = X[~nan_ix]
        y = y[~nan_ix]
        X = np.c_[X[Pclass_ix], X[Sex_ix], X[Cabin_ix],
            X[Embarked_ix], X[SibSp_ix] + X[Parch_ix]]
        self.model = self.model.fit(X,y)
        return self
    def transform(self, X):
        y = X[Fare_ix]
        nan_ix = np.isnan(y.astype(float))
        #select predictor set
        X_pred = np.c_[X[Pclass_ix], X[Sex_ix], X[Cabin_ix],
            X[Embarked_ix], X[SibSp_ix] + X[Parch_ix]]
        X_pred = X_pred[nan_ix]
        #select features
        #replace nan variables
        X[nan_ix, Fare_ix[1]] = self.model.predict(X_pred)
        return X 

class ImputeAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = XGBRegressor(max_depth=3, max_leaf_nodes=4, n_estimators=13)
    def fit(self, X=None, y=None):
        y = X[Age_ix]
        nan_ix = np.isnan(y.astype(float))
        X = X[~nan_ix]
        y = y[~nan_ix]
        X = np.c_[X[Pclass_ix], X[SibSp_ix], X[Parch_ix], 
                X[Sex_ix], X[Fare_ix]]
        self.model = self.model.fit(X, y)
        return self
    def transform(self, X):
        y = X[Age_ix]
        nan_ix = np.isnan(y.astype(float))
        #select predictor set
        X_pred = np.c_[X[Pclass_ix], X[SibSp_ix], X[Parch_ix], 
                X[Sex_ix], X[Fare_ix]]
        X_pred = X_pred[nan_ix]
        #replace nan variables
        X[nan_ix, Age_ix[1]] = self.model.predict(X_pred)
        return X