"""Bootstrap para estimar a variabilidade de uma estatística."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Função que recebe um ou mais arrays e devolve a estatística.
StatisticCallable = Callable[..., Any]


def bootstrap(
    data: ArrayLike | tuple[ArrayLike, ...],
    statistic: StatisticCallable,
    n_bootstrap: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArray[Any], NDArray[Any] | float]:
    """Realiza bootstrap para estimar a variabilidade de uma estatística.

    Parameters
    ----------
    data : array-like ou tupla de array-like
        Dados de entrada. Pode ser um array ou uma tupla de arrays.

    statistic : callable
        Função que calcula a estatística de interesse.

    n_bootstrap : int, default=1000
        Número de reamostragens bootstrap a serem realizados.

    random_state : int, Generator ou None, default=None
        Seed para o gerador de números aleatórios.

    Returns
    -------
    estimates : ndarray
        Estimativas da estatística calculadas a partir das amostras bootstrap.

    std_error : float ou ndarray
        Desvio padrão das estimativas.

    Raises
    ------
    ValueError
        Se n_bootstrap não for inteiro.
        Se nenhum dado for fornecido.
        Se os arrays não tiverem o mesmo tamanho.
        Se o tamanho do dado for 0.
        Se n_bootstrap for menor que 2.
    """

    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError("n_bootstrap deve ser um número inteiro.")

    if not isinstance(data, tuple):
        data = (data,)
    arrays = tuple(np.asarray(d) for d in data)

    if not arrays:
        raise ValueError("Nenhum dado fornecido para bootstrap.")
    n = len(arrays[0])
    if n == 0:
        raise ValueError("O tamanho do dado deve ser maior que 0.")
    if any(len(a) != n for a in arrays):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho.")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap deve ser pelo menos 2.")

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    estimates = np.array([statistic(*(a[rows] for a in arrays)) for rows in idx])
    std_error = np.std(estimates, axis=0, ddof=1)

    return estimates, std_error


def bootstrap_ci(
    data: ArrayLike | tuple[ArrayLike, ...],
    statistic: StatisticCallable,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> tuple[Any, Any, Any]:
    """Calcula uma estatística e seu intervalo de confiança por bootstrap.

    Parameters
    ----------
    data : array-like ou tupla de array-like
        Dados de entrada. Pode ser um array ou uma tupla de arrays.

    statistic : callable
        Função que calcula a estatística de interesse.

    n_bootstrap : int, default=1000
        Número de reamostragens bootstrap a serem realizados.

    level : float, default=0.95
        Nível de confiança do intervalo, estritamente entre 0 e 1.

    random_state : int, Generator ou None, default=None
        Seed para o gerador de números aleatórios.

    Returns
    -------
    estimate : ndarray
        Estatística calculada nos dados originais.

    low : float ou ndarray
        Limite inferior do intervalo de confiança.

    high : float ou ndarray
        Limite superior do intervalo de confiança.

    Raises
    ------
    ValueError
        Se n_bootstrap não for inteiro.
        Se level não for um float estritamente entre 0 e 1.
        Se nenhum dado for fornecido.
        Se os arrays não tiverem o mesmo tamanho.
        Se o tamanho do dado for 0.
        Se n_bootstrap for menor que 2.

    Notes
    -----
    O intervalo é calculado pelo método do percentil: com level=0.95,
    os limites são os percentis 2,5 e 97,5 das estimativas bootstrap.
    Se statistic devolve um array, estimate, low e high têm o mesmo
    formato dele, com um intervalo para cada posição.
    """
    is_frac = isinstance(level, float) and 0 < level < 1
    if not is_frac:
        raise ValueError("level deve ser um float estritamente entre 0 e 1.")

    estimates, _ = bootstrap(data, statistic, n_bootstrap, random_state)

    arrays = data if isinstance(data, tuple) else (data,)
    estimate = statistic(*(np.asarray(a) for a in arrays))

    alpha = 100 * (1 - level) / 2
    low, high = np.percentile(estimates, [alpha, 100 - alpha], axis=0)
    return estimate, low, high
