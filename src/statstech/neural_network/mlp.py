import numpy as np 

from statstech.base import BaseEstimator, RegressorMixin

class MLPRegressor(BaseEstimator, RegressorMixin):
    """docs strings para escrever ainda
    """
    
    def __init__(self, hidden_layer_sizes=(32,), activation="relu",
                 learning_rate=1e-3, max_iter=200, batch_size=32,
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def _forward(self, X):
        raise NotImplementedError
    
    def _backward(self, X, y, activations):
        raise NotImplementedError