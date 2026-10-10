import numpy as np


def train_test_split(X, y, test_size=0.25, shuffle=True,
                     stratify=None, random_state=None):
    """Divide os dados em conjuntos de treino e teste.

    Parameters
    ----------
    X : array-like
        Dados de entrada (features).

    y : array-like
        Rótulos ou valores alvo correspondentes aos dados de entrada. 

    test_size: float ou int, default=0.25
        Se float, representa a proporção do conjunto de teste em relação ao total de dados (número de amostras arredondado para cima).
        Se int, representa o número absoluto de amostras no conjunto de teste.

    shuffle : bool, default=True
        Se True, os dados serão embaralhados antes da divisão.
        Se False, os dados serão divididos na ordem original.

    stratify : array-like ou None, default=None
        Ainda não implementado.
        Se fornecido, levanta NotImplementedError.
        Se None, a divisão é aleatória.  
    
    random_state : int ou None, default = None
        Seed para o gerador de números aleatórios.

    Returns
    ------- 
    X_train : ndarray
        Conjunto de treino (features).
    
    X_test : ndarray
        Conjunto de teste (features).   

    y_train : ndarray   
        Conjunto de treino (rótulos ou valores alvo).

    y_test : ndarray
        Conjunto de teste (rótulos ou valores alvo).

    Raises
    ------
    ValueError
        Se X e y não tiverem o mesmo tamanho.
        Se test_size não for um float entre 0 e 1 ou um inteiro entre 0 e n.
        Se train_size for menor ou igual a 0.
    
    NotImplementedError
        Se stratify for fornecido.
    
    """

    if stratify is not None:
        raise NotImplementedError("Estratificação ainda não implementada.")
    
    X, y = np.asarray(X), np.asarray(y)

    n = len(X)
    if n != len(y):
        raise ValueError(f"X e y devem ter o mesmo tamanho ({n} != {len(y)}).")

    rng = np.random.default_rng(random_state)

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size deve ser um float entre 0 e 1 "
            f" (recebido: {test_size}).")
        n_test = int(np.ceil(n * test_size))
    elif isinstance(test_size, int):
        if not 0 < test_size < n:
            raise ValueError("test_size deve ser um inteiro entre 0 e n "
            f" (recebido: {test_size}).")
        n_test = test_size
    else:
        raise ValueError("test_size deve ser um float ou um inteiro.")

    n_train = n - n_test
    if n_train <= 0:
        raise ValueError("train_size precisa ser pelo menos 1.")

    idx = rng.permutation(n) if shuffle else np.arange(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Dúvida: y opcional para não supervisionado?

class KFold: 
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        pass

    def split(self, X):
        pass

class LeaveOneOut(KFold):
    def split(self, X):
        pass