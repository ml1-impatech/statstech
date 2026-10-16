import numpy as np


class NotFittedError(ValueError, AttributeError):
    """Erro levantado quando o modelo é usado antes do fit."""


def _is_numeric(a):
    """Diz se o array é numérico (inteiro, float ou bool)."""
    return np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)


def _check_finite(a, name, allow_nan=False):
    """Erro se um array tiver NaN ou algum valor infinito."""
    if allow_nan:
        if np.isinf(a).any():
            raise ValueError(f"{name} contém valores infinitos.")
    elif not np.isfinite(a).all():
        raise ValueError(f"{name} contém NaN ou valores infinitos.")


def check_array(X, allow_nan=False, ensure_numeric=True):
    """Verifica se X é um array 2D.

    Converte X para array, confere se é 2D, se não está vazio, se é
    numérico e se não tem NaN ou algum valor infinito.

    Parameters
    ----------
    X : array-like de formato (n_amostras, n_features)
        Dado de entrada a ser verificado.
    allow_nan : bool, default=False
        Se True, aceita NaN, mas valores infinitos continuam proibidos.
    ensure_numeric : bool, default=True
        Se True, exige que X seja numérico. False é usado para dados
        categóricos.

    Returns
    -------
    X : ndarray de formato (n_amostras, n_features)
        X convertido para array do NumPy 2D.

    Raises
    ------
    ValueError
        Se X não for 2D, estiver vazio, não for numérico no caso
        ensure_numeric=True, tiver NaN, no caso allow_nan=False,
        ou valores infinitos.
    """
    X = np.asarray(X)
    n = X.ndim
    if n == 1:
        raise ValueError(
            "Esse array é 1D, precisa ser 2D. Sugestão: "
            "usar X.reshape(-1, 1) se for uma feature só"
        )
    if n == 0 or n > 2:
        raise ValueError(f"O array não é 2D, tem {X.ndim} dimensões")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"O array está vazio, formato {X.shape}")
    if _is_numeric(X):
        _check_finite(X, "X", allow_nan)
    elif ensure_numeric:
        raise ValueError(f"X tem dtype {X.dtype}, precisa ser numérico")
    return X


def check_X_y(X, y):
    """
    Verifica se X e y são arrays válidos e se têm o mesmo tamanho.

    X precisa ser um array 2D numérico, y precisa ser 1D e ter o mesmo
    número de amostras que X. y pode ser numérico ou texto para
    classificação.

    Parameters
    ----------
    X : array-like de formato (n_amostras, n_features)
        Dado de entrada a ser verificado.
    y : array-like de formato (n_amostras,)
        Resultado numérico ou em texto a ser verificado.

    Returns
    -------
    X : ndarray de formato (n_amostras, n_features)
        X convertido para array.
    y : ndarray de formato (n_amostras,)
        y convertido para array.

    Raises
    ------
    ValueError
        Se X for inválido, se y não for 1D, se X e y tiverem números
        de amostras diferentes ou se y tiver NaN, None ou valores
        infinitos.
    """
    X = check_array(X)
    y = np.asarray(y)
    n = y.ndim
    if n == 2 and y.shape[1] == 1:
        raise ValueError(
            "y tem formato (n, 1), mas precisa ser 1D. Sugestão: use y.ravel()"
        )
    if n != 1:
        raise ValueError(f"y deve ser 1D, mas tem {y.ndim} dimensões")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X e y têm tamanhos diferentes: {X.shape[0]} e {y.shape[0]}")
    if _is_numeric(y):
        _check_finite(y, "y")
    elif y.dtype == object and any(
        v is None or (isinstance(v, float) and np.isnan(v)) for v in y
    ):
        raise ValueError("y contém NaN ou None")
    return X, y


def check_is_fitted(model, attributes=None):
    """Verifica se o modelo já foi treinado.

    Se não for passada uma lista de atributos, verifica os atributos
    terminados com _, que é o padrão para todos os atributos criados no
    fit. Se houver uma lista de atributos, é feita a verificação de que
    todos os atributos da lista existem.

    Parameters
    ----------
    model : objeto estimador
        Modelo a ser verificado.
    attributes : str ou lista de str, default=None
        Atributos que o fit precisa ter criado.

    Raises
    ------
    NotFittedError
        Se o modelo ainda não tiver sido treinado.
    """
    name = type(model).__name__
    if attributes is None:
        fitted = [a for a in vars(model) if a.endswith("_") and not a.startswith("__")]
        if not fitted:
            raise NotFittedError(f"{name} ainda não foi treinado")
        return
    if isinstance(attributes, str):
        attributes = [attributes]
    missing = [a for a in attributes if not hasattr(model, a)]
    if missing:
        raise NotFittedError(f"{name} ainda não foi treinado, faltam {missing}")
