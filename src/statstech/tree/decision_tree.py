"""Árvores de decisão (CART) para regressão e classificação."""

from dataclasses import dataclass

import numpy as np

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: object = None  # só nas folhas


# --- impurezas (funções soltas, testáveis sozinhas) ---
def _rss(y):
    pass


def _gini(y):
    pass


def _entropy(y):
    pass


class _BaseDecisionTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _build(self, X, y, depth):
        pass

    def _best_split(self, X, y):
        pass

    def _impurity(self, y):  # cada subclasse define
        pass

    def _leaf_value(self, y):  # cada subclasse define
        pass

    def _predict_one(self, x):
        pass


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        pass

    def fit(self, X, y):  # guarda classes_ e converte y para índices
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass
