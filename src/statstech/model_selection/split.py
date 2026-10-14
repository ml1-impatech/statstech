import numpy as np


def _resolve_n_test(test_size, n):
    """Converte test_size no número de amostras do conjunto de teste."""
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError(
                f"test_size deve ser um float entre 0 e 1 (recebido: {test_size})."
            )
        n_test = int(np.ceil(n * test_size))
    elif isinstance(test_size, int):
        if not 0 < test_size < n:
            raise ValueError(
                f"test_size deve ser um inteiro entre 0 e n={n} "
                f"(recebido: {test_size})."
            )
        n_test = test_size
    else:
        raise ValueError("test_size deve ser um float ou um inteiro.")

    if n - n_test <= 0:
        raise ValueError("train_size precisa ser pelo menos 1.")
    return n_test


def _stratified_indices(stratify, n_test, rng):
    """Sorteia índices de treino e teste preservando as proporções."""
    n = len(stratify)
    train_parts, test_parts = [], []

    for k in np.unique(stratify):
        class_idx = np.flatnonzero(stratify == k)
        if len(class_idx) < 2:
            raise ValueError(
                f"A classe {k} tem apenas {len(class_idx)} "
                "amostra(s); são necessárias pelo menos 2."
            )

        class_idx = rng.permutation(class_idx)
        n_test_k = round(len(class_idx) * n_test / n)
        n_test_k = max(1, min(n_test_k, len(class_idx) - 1))

        test_parts.append(class_idx[:n_test_k])
        train_parts.append(class_idx[n_test_k:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return train_idx, test_idx


def train_test_split(
    X, y, test_size=0.25, shuffle=True, stratify=None, random_state=None
):
    """Divide os dados em conjuntos de treino e teste.

    Parameters
    ----------
    X : array-like
        Dados de entrada (features).

    y : array-like
        Rótulos ou valores alvo correspondentes aos dados de entrada.

    test_size : float ou int, default=0.25
        Se float, representa a proporção do conjunto de teste
        em relação ao total de dados (número de amostras arredondado para cima).
        Se int, representa o número absoluto de amostras no conjunto de teste.

    shuffle : bool, default=True
        Se True, os dados serão embaralhados antes da divisão.
        Se False, os dados serão divididos na ordem original.

    stratify : array-like ou None, default=None
        Rótulos usados para manter, no treino e no teste, a mesma
        proporção de cada classe
        dos dados originais (normalmente é o próprio y).
        Exige shuffle=True. Com stratify, o tamanho do conjunto de teste
        é aproximado. Se None, a divisão é aleatória.

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
        Se test_size não for um float entre 0 e 1 ou um inteiro
        entre 0 e n.
        Se train_size for menor ou igual a 0.
        Se stratify não tiver o mesmo tamanho de X.
        Se stratify for fornecido com shuffle=False.
        Se alguma classe de stratify tiver menos de 2 amostras.
    """
    X, y = np.asarray(X), np.asarray(y)

    n = len(X)
    if n != len(y):
        raise ValueError(f"X e y devem ter o mesmo tamanho ({n} != {len(y)}).")

    rng = np.random.default_rng(random_state)
    n_test = _resolve_n_test(test_size, n)

    if stratify is not None:
        stratify = np.asarray(stratify)
        if len(stratify) != n:
            raise ValueError(
                f"X e stratify devem ter o mesmo tamanho ({n} != {len(stratify)})."
            )
        if not shuffle:
            raise ValueError("stratify exige shuffle=True.")
        train_idx, test_idx = _stratified_indices(stratify, n_test, rng)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        train_idx, test_idx = idx[: n - n_test], idx[n - n_test :]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class KFold:
    """Particiona o conjunto de dados em k folds de tamanhos proporcionais.

    Para cada partição, uma é usada como conjunto de teste
    e k-1 são usadas como conjunto de treino.

    Parameters
    ----------
    n_splits : int, default=5
        Quantidade de partições (k). Deve ser pelo menos 2.

    shuffle : bool, default=False
        Se True, os dados serão embaralhados antes da divisão.
        Se False, os dados serão divididos na ordem original.

    random_state : int ou None, default=None
        Seed para o gerador de números aleatórios.

    Raises
    ------
    ValueError
        Se n_splits for menor que 2.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits deve ser pelo menos 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """Gera os índices de treino e teste de cada partição.

        Para cada uma das n_splits partições, devolve os índices
        das amostras de treino e de teste.
        Cada amostra aparece no conjunto de teste exatamente uma vez.

        Parameters
        ----------
        X : array-like
            Dados de entrada (features). Apenas os índices são utilizados.

        Yields
        ------
        train_idx : ndarray
            Índices das amostras de treino da partição.

        test_idx : ndarray
            Índices das amostras de teste da partição.

        Raises
        ------
        ValueError
            Se a quantidade de folds for menor que 2.
            Se n_splits for maior que n.
        """
        if self.n_splits < 2:
            raise ValueError("n_splits deve ser pelo menos 2.")

        n = len(X)
        if self.n_splits > n:
            raise ValueError("n_splits não deve ser maior que n.")

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n) if self.shuffle else np.arange(n)
        folds = np.array_split(idx, self.n_splits)

        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate(folds[:i] + folds[i + 1 :])
            yield train_idx, test_idx


class LeaveOneOut(KFold):
    """Validação cruzada deixando uma amostra de fora (k = n).

    Em cada rodada, uma única amostra é usada como teste e as
    n - 1 restantes são usadas como treino, gerando n partições.

    Parameters
    ----------
    n_splits, shuffle, random_state
        Herdados de KFold, mas ignorados: o número de partições é
        sempre o número de amostras e a ordem nunca é embaralhada.
    """

    def split(self, X):
        """Gera os índices de treino e teste de cada partição.

        Parameters
        ----------
        X : array-like
            Dados de entrada. Apenas o número de amostras é utilizado.

        Yields
        ------
        train_idx : ndarray
            Índices das n - 1 amostras de treino da partição.
        test_idx : ndarray
            Índice da única amostra de teste da partição.

        Raises
        ------
        ValueError
            Se X tiver menos de 2 amostras.
        """
        if len(X) < 2:
            raise ValueError(
                f"LeaveOneOut precisa de pelo menos 2 amostras (recebido {len(X)})."
            )
        yield from KFold(n_splits=len(X)).split(X)
