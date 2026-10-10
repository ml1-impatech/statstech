import numpy as np

def bootstrap(data, statistic, n_bootstrap=1000,
              random_state=None):
    """Realiza bootstrap para estimar a variabilidade de uma estatística.

    Parameters
    ----------
    data : array-like ou tupla de array-like
        Dados de entrada. Pode ser um array ou uma tupla de arrays.
        
    statistic : callable
        Função que calcula a estatística de interesse.
    
    n_bootstrap : int, default=1000
        Número de reamostragens bootstrap a serem realizados.

    random_state : int ou None, default = None
        Seed para o gerador de números aleatórios.

    Returns
    -------
    estimates : ndarray
        Estimativas da estatística calculadas a partir das amostras bootstrap.

    std_error : float ou ndarray
        Desvio padrão das estimativas. 
    """
    # Verifica se os dados são uma tupla, caso contrário, converte para tupla
    if not isinstance(data, tuple):
        data = (data,)
    arrays = tuple(np.asarray(d) for d in data)

    if not arrays:
        raise ValueError("Nenhum dado fornecido para bootstrap.")
    n = len(arrays[0])
    if n==0:
        raise ValueError("O tamanho do dado deve ser maior que 0.")
    if any(len(a) != n for a in arrays):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho.")    
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap deve ser pelo menos 2.")

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    estimates = np.array([statistic(*(a[rows] for a in arrays)) for rows in idx])
    std_error = np.std(estimates, axis = 0, ddof = 1)

    return estimates, std_error

# Dúvida: retornar intervalo de confiança também? 