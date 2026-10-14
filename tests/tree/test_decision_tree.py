"""Decision trees (CART) for regression and classification."""

from dataclasses import dataclass

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None  # only in leaves


class _BaseDecisionTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _build(self, X, y, depth):  # recursive
        pass

    def _impurity(self, y):  # each subclass defines
        pass

    def _leaf_value(self, y):  # each subclass defines
        pass


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def _impurity(self, y):
        pass

    def _leaf_value(self, y):
        pass