import numpy as np
import numpy.typing as npt

def _validate_bin_roc_auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike):
    """Valida as entradas de binary_roc_auc.

    Parameters
    ----------
    y_true : array-like
        Array unidimensional com os rótulos reais das observações,
        contendo apenas 0 e 1.
    y_score : array-like
        Array unidimensional com os scores atribuídos pelo modelo à
        classe positiva. Scores maiores indicam maior evidência da
        classe positiva.

    Returns
    -------
    y_true : np.ndarray
        Array de rótulos reais validado e transformado em array NumPy.
    y_score : np.ndarray
        Array de scores validado e transformado em array NumPy.

    Raises
    ------
    ValueError
        Se os arrays não forem unidimensionais, tiverem tamanhos
        diferentes ou estiverem vazios.
        Se y_true contiver rótulos diferentes de 0 e 1 ou se y_score
        contiver valores não finitos.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true e y_score devem ser unidimensionais.")

    if y_true.size != y_score.size:
        raise ValueError("y_true e y_score devem ter o mesmo tamanho.")

    if y_true.size == 0:
        raise ValueError("y_true e y_score devem ser não vazios.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true deve conter apenas 0 e 1.")

    _check_finite(y_score, "y_score")

    return y_true, y_score

def _validate_roc_auc_score(y_true: npt.ArrayLike, y_score: npt.ArrayLike, class_k: int | None = None) -> float:
    """Valida os parâmetros de roc_auc_score.

    Parameters
    ----------
    y_true: array-like
        Array unidimensional com os rótulos reais das observações.
        Na classificação binária, deve conter apenas 0 e 1.
        Na classificação multiclasse, deve conter rótulos correspondentes
        aos índices das classes entre 0 e n_classes - 1.
    y_score: array-like
        Na classificação binária, array unidimensional com os scores
        da classe positiva.
        Na classificação multiclasse, matriz de dimensões
        (n_samples, n_classes), na qual cada elemento [i][j]
        representa o score atribuído à classe j para a observação i.
        A coluna j deve corresponder ao rótulo j.
    class_k: int or None, default None
        Índice da classe cuja área ROC-AUC será calculada na
        classificação multiclasse, considerando essa classe como
        positiva e todas as demais como negativas (One-vs-Rest).
        Deve ser None na classificação binária e um índice válido
        na classificação multiclasse.

    Returns
    -------
    auc: float
        Resultado de binary_roc_auc, quando y_score for unidimensional.
    y_true: np.ndarray
        Array de rótulos reais validado e transformado em array NumPy,
        quando y_score for bidimensional.
    y_score: np.ndarray
        Matriz de scores validada e transformada em array NumPy,
        quando y_score for bidimensional.

    Raises
    ------
    ValueError
        Se as dimensões ou os tamanhos dos arrays forem inválidos,
        se y_score contiver valores não finitos ou se houver menos
        de duas colunas na matriz multiclasse.
        Se class_k for incompatível com o tipo de classificação,
        não for um inteiro válido ou estiver fora do intervalo das
        classes.
        Se y_true contiver rótulos que não correspondam aos índices
        das colunas de y_score na classificação multiclasse.
        Também pode ser gerado se os dados da classificação binária
        forem inválidos.
    """
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1: 
        raise ValueError("y_true deve ser unidimensional.")

    if y_score.ndim == 1:
        if class_k is not None:
            raise ValueError("class_k deve ser None para classificação binária.")

    if y_score.ndim != 2:
        raise ValueError("y_score deve ser unidimensional ou bidimensional.")

    if y_true.size != y_score.shape[0]:
        raise ValueError("O tamanho de y_score deve ser o número de linhas de y_true.")

    if y_score.shape[1] < 2:
        raise ValueError("y_score deve conter ao menos 2 colunas de classes.")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Os valores de y_score devem ser finitos.")

    if class_k is None:
        raise ValueError("class_k é exigida para classificações multiclasse.")

    if isinstance(class_k, (bool, np.bool_)) or not isinstance(class_k, (int, np.integer)):
        raise ValueError("class_k deve ser um inteiro.")

    n_classes = y_score.shape[1]

    if class_k < 0 or class_k >= n_classes:
        raise ValueError("class_k está fora do intervalo válido.")

    if not np.all(np.isin(y_true, np.arange(n_classes))):
        raise ValueError("Os labels de y_true devem corresponder aos índices das colunas de y_score.")

    return y_true, y_score

def _validate_log_loss_entries(y_true: npt.ArrayLike, p_matrix: npt.ArrayLike, classes: list[int], epsilon: float = 1e-10):
    """Valida as entradas de log_loss.

    Parameters
    ----------
    y_true : array-like
        Array unidimensional com os rótulos reais das observações.
    p_matrix : array-like
        Matriz de dimensões (n_samples, n_classes), na qual cada elemento
        na posição [i][j] representa a probabilidade de a observação i
        ser prevista como a classe correspondente à posição j em classes.
    classes : list[int]
        Lista de rótulos das classes, sem duplicatas, cujo número deve
        ser igual ao número de colunas de p_matrix.
    epsilon : float, default=1e-10
        Número utilizado para evitar que se tenha log de zero.

    Returns
    -------
    y_true : np.ndarray
        Array de rótulos reais validado e transformado em array NumPy.
    p_matrix : np.ndarray
        Matriz de probabilidades validada e transformada em array NumPy.
    classes : list[int]
        Lista de rótulos das classes fornecida como entrada.

    Raises
    ------
    ValueError
        Se as dimensões ou os tamanhos dos arrays forem inválidos.
        Se y_true contiver NaN, None ou valores infinitos.
        Se o número de classes for diferente do número de colunas de
        p_matrix ou se classes contiver duplicatas.
        Se epsilon não for finito ou não estiver entre 0 e 1.
        Se p_matrix contiver valores não finitos, probabilidades fora
        do intervalo [0, 1] ou linhas cujas probabilidades não somem 1.
    """
    p_matrix = np.asarray(p_matrix, dtype=np.float64)
    y_true = np.asarray(y_true)

    p_matrix, y_true = check_X_y(p_matrix, y_true)

    n_samples, n_classes = p_matrix.shape

    if len(classes) != n_classes:
        raise ValueError(
            "O número de classes deve ser igual ao número de colunas de p_matrix."
        )

    if len(set(classes)) != len(classes):
        raise ValueError("classes não pode conter duplicatas.")

    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon deve ser finito e estar entre 0 e 1.")

    if np.any(p_matrix < 0) or np.any(p_matrix > 1):
        raise ValueError("p_matrix contém probabilidades fora do intervalo.")

    if not np.allclose(p_matrix.sum(axis=1), 1.0):
        raise ValueError("As probabilidades de cada linha devem somar 1.")

    return y_true, p_matrix, classes

def _validate_confusion_matrix(c_matrix: list[list[int]]) -> None:
    """Valida se a matriz de confusão é não vazia e quadrada.

    Parameters
    ----------
    c_matrix : list[list[int]]
        Matriz de confusão a ser validada.

    Returns
    -------
    None
        A função não retorna nenhum valor.

    Raises
    ------
    ValueError
        Se c_matrix não for bidimensional, estiver vazia, não for
        numérica, contiver valores não finitos ou não for quadrada.
    """
    c_matrix = check_array(c_matrix)

    if c_matrix.shape[0] != c_matrix.shape[1]:
        raise ValueError("A matriz de confusão deve ser quadrada.")

def _validate_class_index(c_matrix: list[list[int]], class_k: int) -> None:
    """Valida a matriz de confusão e o índice da classe.

    Parameters
    ----------
    c_matrix : list[list[int]]
        Matriz de confusão cuja estrutura será validada.
    class_k : int
        Índice da classe a ser validado, começando em zero.

    Returns
    -------
    None
        A função não retorna nenhum valor.

    Raises
    ------
    ValueError
        Se c_matrix não for bidimensional, estiver vazia, não for
        numérica, contiver valores não finitos ou não for quadrada.
        Se class_k não for um inteiro válido ou estiver fora do
        intervalo dos índices das classes da matriz de confusão.
    """
    _validate_confusion_matrix(c_matrix)

    if isinstance(class_k, (bool, np.bool_)) or not isinstance(
        class_k, (int, np.integer)
    ):
        raise ValueError("class_k deve ser um inteiro.")

    if not 0 <= class_k < len(c_matrix):
        raise ValueError("class_k está fora do intervalo válido.")


def _validate_regression_inputs(
        y_true: npt.ArrayLike, 
        y_pred: npt.ArrayLike,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Valida e converte os arrays de valores reais e previstos.
    
    Parameters
    ----------
    y_true: array-like
        Array com os valores reais do alvo.
    y_pred: array-like
        Array com os valores previstos pelo modelo.


    Returns
    -------
    y_true: numpy.ndarray
        Array unidimensional validado, com os valores float64 reais. 
    y_pred: numpy.ndarray
        Array unidimensional validado, com os valores float64 previstos. 

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("Os arrays devem ter o mesmo formato.")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Os arrays devem ser unidimensionais.")

    if y_pred.size == 0:
        raise ValueError("Os arrays devem não estar vazios.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Os arrays não podem conter NaN ou infinito.")

    return y_true, y_pred

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
