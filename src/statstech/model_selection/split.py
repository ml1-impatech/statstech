import numpy as np


def train_test_split(X, y, test_size=0.25, shuffle=True,
                     stratify=None, random_state=None):
    if stratify is not None:
        raise NotImplementedError("Estratificação ainda não implementada.")
    
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)
    if n != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho.")

    rng = np.random.default_rng(random_state)

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size deve ser um float entre 0 e 1.")
        n_test = n*test_size
        n_test = int(np.ceil(n_test))
    elif isinstance(test_size, int):
        if not 0 < test_size < n:
            raise ValueError("test_size deve ser um inteiro entre 0 e n.")
        n_test = test_size
    else:
        raise ValueError("test_size deve ser um float ou um inteiro.")
    if n-n_test <= 0:
        raise ValueError("train_size precisa ser pelo menos 1.")

    if shuffle:
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)

    X_train = X[idx[:-n_test]]
    y_train = y[idx[:-n_test]]
    X_test = X[idx[-n_test:]]   
    y_test = y[idx[-n_test:]]

    return X_train, X_test, y_train, y_test

class KFold: 
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        pass

    def split(self, X):
        pass

class LeaveOneOut(KFold):
    def split(self, X):
        pass