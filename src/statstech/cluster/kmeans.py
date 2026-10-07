import numpy as np
from statstech.base import BaseEstimator

class Kmeans(BaseEstimator):

    def __init__(self, n_clusters=8, n_init=10, max_iter=300,
                 tol=1e-4, init="k-means++", random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X, y=None):
        raise self.fit(X).labels_

    

    