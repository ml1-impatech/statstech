"""Divisão em treino e teste, KFold e LeaveOneOut."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _resolve_n_test(test_size: float, n: int) -> int:
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
        raise TypeError("test_size deve ser um float ou um inteiro.")

    if n - n_test <= 0:
        raise ValueError("train_size precisa ser pelo menos 1.")
    return n_test


def _stratified_indices(
    stratify: NDArray[Any], n_test: int, rng: np.random.Generator
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
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
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Divide os dados em conjuntos de treino e teste.

    Parameters
    ----------
    X : array-like de formato (n_amostras, n_features)
        Dados de entrada. Aceita qualquer tipo de dado,
        inclusive texto e NaN.
        Para uma única feature, use X.reshape(-1, 1).

    y : array-like de formato (n_amostras,)
        Valores alvo. Deve ser 1D e ter o mesmo número de amostras
        de X.

    test_size : float ou int, default=0.25
        Se float, representa a proporção do conjunto de teste em relação ao
        total de dados (número de amostras arredondado para cima).
        Se int, representa o número absoluto de amostras no conjunto de teste.

    shuffle : bool, default=True
        Se True, os dados serão embaralhados antes da divisão.
        Se False, os dados serão divididos na ordem original.

    stratify : array-like ou None, default=None
        Rótulos usados para manter, no treino e no teste, a mesma proporção
        de cada classe dos dados originais (normalmente é o próprio y).
        Exige shuffle=True. Com stratify, o tamanho do conjunto de teste
        é aproximado. Se None, a divisão é aleatória.

    random_state : int, Generator ou None, default=None
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
        Se X ou y forem inválidos: X que não seja 2D, y que não
        seja 1D, ou tamanhos diferentes.
        Se test_size não for um float entre 0 e 1 ou um inteiro
        entre 0 e n.
        Se train_size for menor ou igual a 0.
        Se stratify não tiver o mesmo tamanho de X.
        Se stratify for fornecido com shuffle=False.
        Se alguma classe de stratify tiver menos de 2 amostras.
    """
    X_arr, y_arr = np.asarray(X), np.asarray(y)

    if X_arr.ndim != 2:
        raise ValueError("X precisa ser 2D (n_amostras, n_features).")
    if y_arr.ndim != 1:
        raise ValueError("y deve ser 1D (n_amostras,).")
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"X e y devem ter o mesmo tamanho ({len(X_arr)} != {len(y_arr)})."
        )
    n = len(X_arr)

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

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


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

    random_state : int, Generator ou None, default=None
        Semente do gerador aleatório. Use um inteiro para obter os
        mesmos folds a cada chamada de split; com um Generator, cada
        chamada consome o gerador e devolve folds diferentes.

    Raises
    ------
    ValueError
        Se n_splits for menor que 2 ou se não for inteiro.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
            raise ValueError("n_splits deve ser um número inteiro.")
        if n_splits < 2:
            raise ValueError("n_splits deve ser pelo menos 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Gera os índices de treino e teste de cada partição.

        Para cada uma das n_splits partições, devolve os índices
        das amostras de treino e de teste.
        Cada amostra aparece no conjunto de teste exatamente uma vez.

        Parameters
        ----------
        X : array-like
            Dados de entrada (features). Apenas os índices são utilizados.

        y : array-like de formato (n_amostras,), default=None
            Ignorado. Existe para manter a mesma interface do
            StratifiedKFold.

        Yields
        ------
        train_idx : ndarray
            Índices das amostras de treino da partição.

        test_idx : ndarray
            Índices das amostras de teste da partição.

        Raises
        ------
        ValueError
            Se n_splits for maior que n.
        """

        n = len(np.asarray(X))
        if self.n_splits > n:
            raise ValueError("n_splits não deve ser maior que n.")

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n) if self.shuffle else np.arange(n)
        folds = np.array_split(idx, self.n_splits)

        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate(folds[:i] + folds[i + 1 :])
            yield train_idx, test_idx


class LeaveOneOut:
    """Validação cruzada deixando uma amostra de fora (k = n).

    Em cada rodada, uma única amostra é usada como teste e as
    n - 1 restantes são usadas como treino, gerando n partições.
    Não tem parâmetros: o número de partições é o número de amostras.
    """

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Gera os índices de treino e teste de cada partição.

        Parameters
        ----------
        X : array-like
            Dados de entrada. Apenas o número de amostras é utilizado.

        y : array-like de formato (n_amostras,), default=None
            Ignorado. Existe para manter a mesma interface do
            StratifiedKFold.

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
        n = len(np.asarray(X))
        n = len(np.asarray(X))
        if n < 2:
            raise ValueError(
                f"LeaveOneOut precisa de pelo menos 2 amostras (recebido {n})."
            )
        yield from KFold(n_splits=n).split(X)


class StratifiedKFold(KFold):
    """K folds que mantêm a proporção de cada classe de y.

    Funciona como o KFold, mas cada classe é dividida igualmente
    entre os folds. Cada classe precisa ter pelo menos n_splits
    exemplos.

    Parameters
    ----------
    n_splits : int, default=5
        Quantidade de partições (k). Deve ser pelo menos 2.

    shuffle : bool, default=False
        Se True, os dados serão embaralhados antes da divisão.
        Se False, os dados serão divididos na ordem original.

    random_state : int, Generator ou None, default=None
        Semente do gerador aleatório. Use um inteiro para obter os
        mesmos folds a cada chamada de split; com um Generator, cada
        chamada consome o gerador e devolve folds diferentes.

    Raises
    ------
    ValueError
        Se n_splits for menor que 2 ou se não for inteiro.
    """

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Gera os índices de treino e teste de cada partição.

        Para cada uma das n_splits partições, devolve os índices
        das amostras de treino e de teste.
        Cada amostra aparece no conjunto de teste exatamente uma vez.

        Parameters
        ----------
        X : array-like
            Dados de entrada (features). Apenas os índices são utilizados.

        y : array-like de formato (n_amostras,)
            Classes de cada amostra, usadas para estratificar. Obrigatório.

        Yields
        ------
        train_idx : ndarray
            Índices das amostras de treino da partição.

        test_idx : ndarray
            Índices das amostras de teste da partição.

        Raises
        ------
        ValueError
            Se y for None.
            Se X e y não tiverem o mesmo tamanho.
            Se alguma classe de y tiver menos que n_splits exemplos.
        """
        if y is None:
            raise ValueError("StratifiedKFold precisa de y.")
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X e y devem ter o mesmo tamanho ({len(X)} != {len(y)}).")

        classes, counts = np.unique(y, return_counts=True)
        for label, count in zip(classes, counts):
            if count < self.n_splits:
                raise ValueError(
                    f"A classe {label!r} tem só {count} exemplo(s); "
                    f"são necessários pelo menos n_splits={self.n_splits}."
                )

        rng = np.random.default_rng(self.random_state)

        pieces = []
        for label in classes:
            idx = np.flatnonzero(y == label)
            if self.shuffle:
                idx = rng.permutation(idx)
            pieces.append(np.array_split(idx, self.n_splits))

        for i in range(self.n_splits):
            test_idx = np.concatenate([p[i] for p in pieces])
            mask = np.ones(len(y), dtype=bool)
            mask[test_idx] = False
            yield np.flatnonzero(mask), test_idx
