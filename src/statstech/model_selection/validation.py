import numpy as np

from ..base import clone
from .split import KFold


def _fit_and_score(estimator, X, y, train_idx, test_idx, scoring):
    """Treina uma cópia do modelo e pontua no conjunto de teste.

    Clona o estimador, realiza o treino nos índices especificados e avalia o desempenho usando a 
    métrica padrão ou uma dada.

    Parameters
    ----------
    estimator : objeto com fit, predict e score
        Modelo a ser avaliado.

    X : array-like de shape (n_amostras, n_features)
        Dados de entrada (features). 

    y : array-like de shape (n_amostras,)
        Rótulos ou valores correspondentes a X.

    train_idx : array-like de int ou bool
        Índices ou máscara booleana para filtrar
        as amostras de treino em X e y.

    test_idx : array-like de int ou bool
        Índices ou máscara booleana para filtrar
        as amostras de teste em X e y.

    scoring : callable ou None
        Função com assinatura scoring(y_true, y_pred) que devolve 
        um número. Se None, usa o método score do estimator. 

    Returns
    -------
    score : float
        A pontuação do modelo avaliado nos dados de teste.
    """
    model = clone(estimator)
    model.fit(X[train_idx], y[train_idx])

    X_test, y_test = X[test_idx], y[test_idx]
    if scoring is None:
        return model.score(X_test, y_test)
    return scoring(y_test, model.predict(X_test))


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """Avalia um modelo por validação cruzada.

    Para cada partição gerada por cv, treina uma cópia do modelo
    com os dados de treino e calcula o score nos dados de teste.
    O estimator original não é modificado nem treinado.

    Parameters
    ----------
    estimator : objeto com fit, predict e score
        Modelo a ser avaliado. Uma cópia nova é criada com clone
        a cada partição. 

    X : array-like de shape (n_amostras, n_features)
        Dados de entrada (features).

    y : array-like de shape (n_amostras,)
        Rótulos ou valores correspondentes a X.

    cv : int ou objeto divisor, default=5
        Se int, usa KFold com esse número de partições. Também 
        aceita um divisor já configurado (KFold, LeaveOneOut) que
        tenha o método split.

    scoring : callable ou None, default=None
        Função com assinatura scoring(y_true, y_pred) que devolve 
        um número. Se None, usa o método score do estimator.
    
    Returns
    -------
    scores : ndarray de shape (n_partições,)
        Score obtido no conjunto de teste de cada partição.

    Raises
    ------
    ValueError
        Se X e y não tiverem o mesmo tamanho.
        Se scoring não for None ou callable.
    """
    if scoring is not None and not callable(scoring):
        raise ValueError("scoring deve ser None ou uma função.")
    
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError(
            f"X e y devem ter o mesmo tamanho ({len(X)} != {len(y)}).")
    
    splitter = KFold(n_splits=cv) if isinstance(cv, int) else cv

    return np.array([
        _fit_and_score(estimator, X, y, train_idx, test_idx, scoring)
        for train_idx, test_idx in splitter.split(X)
    ])