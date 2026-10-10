"""Base classes and functions that all methods use."""

import copy
import inspect
import numpy as np

class BaseEstimator:
    """Base-class of all models and transformers.
    
            Requires that all __init__ stores each received parameter in an
            attribute with the same name (self.alpha = alpha). 
            """

    @classmethod
    def _get_param_names(cls):
        """Names of the parameters of __init__"""

        init = cls.__init__

        if init is object.__init__:
            return []
        
        signature = inspect.signature(init)
        names = []

        for p in signature.parameters.values():
            if p.name == "self" or p.kind == p.VAR_KEYWORD:
                continue
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__}: __init__ can't use *args."
                )
            names.append(p.name)

        return sorted(names)
    
    def get_params(self):
        """Returns a dict {hyperparameter_name: value}.""" 

        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params): 
        """"Alters hyperparameters e returns self."""

        valid_params = self._get_param_names()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid Parameter '{key}' for the estimator {type(self).__name__}. "
                    f"Valid Parameters are: {list(valid_params)}"
                )
            setattr(self, key, value)

        return self

    def __repr__(self):

        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class RegressorMixin: 
    pass


class ClassifierMixin: 

    def score(self, X, y):
        """" Retorna a acurácia da classificação. """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class TransformerMixin: 

    def fit_transform(self, X, y=None):
        """ Ajusta e devolve os dados transformados. """
        return self.fit(X,y).transform(X)


def clone(estimator):
    """Creates a new model, untrained, with the same hyperparameters"""

    if isinstance(estimator, (list, tuple)):
        return type(estimator)(clone(e) for e in estimator)
    
    if not hasattr(estimator, "get_params"):
        raise TypeError(
            f"Not possible to clone {estimator!r}: no get_params."
        )

    params = {
        name: clone(value) if hasattr(value, "get_params")
        else copy.deepcopy(value)
        for name, value in estimator.get_params().items()
    }

    return type(estimator)(**params)