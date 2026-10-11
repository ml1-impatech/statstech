import numpy as np
from ..base import clone
from .split import KFold

def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """Avalia um modelo por validação cruzada.

    Para cada partição gerada por cv, treina uma cópia do modelo
    com os dados de treino e calcula o score nos dados de teste.
    O estimator original não é modificado nem treinado.

    Parameters
    ----------
    estimator : objeto com fit, predictor e score
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
    scores : array
        Score obtido no conjunto de teste de cada partição.

    Raises
    ------
    ValueError
        Se X e y não tiverem o mesmo tamanho.
        Se scoring não for None ou callable.
    """
    X, y = np.asarray(X), np.asarray(y)

    if len(X) != len(y):
        raise ValueError(
            f"X e y devem ter o mesmo tamanho ({len(X)} != {len(y)}).")
    
    if isinstance(cv, int):
        splitter = KFold(n_splits=cv)
    else:
        splitter = cv

    if scoring is not None and not callable(scoring):
        raise ValueError("scoring deve ser None ou uma função.")

    scores = []
    for train_idx, test_idx in splitter.split(X):
        copy_model = clone(estimator)
        copy_model.fit(X[train_idx], y[train_idx])

        if scoring is None:
            score = copy_model.score(X[test_idx], y[test_idx])
        else:
            y_pred = copy_model.predict(X[test_idx])
            score = scoring(y[test_idx], y_pred)

        scores.append(score)

    return np.array(scores)

# As métricas precisam ser implementadas 

