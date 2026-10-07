import numpy as np

from statstech.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        raise NotImplementedError