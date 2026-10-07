import numpy as np

def bootstrap(data, statistic, n_bootstrap=1000,
              random_state=None):

    if isinstance(data, tuple):
        arrays = []
        for d in data:
            arrays.append(np.asarray(d))
        for a in arrays:
            if len(a) != len(arrays[0]):
                raise ValueError("Todas as entradas do array devem ter o mesmo comprimento.")
        data = tuple(arrays)
        n = len(data[0])
    else: 
        data =np.asarray(data)
        n = len(data)
        
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap deve ser pelo menos 2.")

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    estimates = []

    for i in range(n_bootstrap):
            if isinstance(data, tuple):
                sample = tuple(d[idx[i]] for d in data)
                estimates.append(statistic(*sample)) # *sample desempacota a tupla para passar como argumentos separados para a função estatística
            else:
                sample = data[idx[i]]
                estimates.append(statistic(sample))

    estimates = np.array(estimates)
    std_error = np.std(estimates, axis = 0, ddof = 1) # ddof: delta degrees of freedom, 1 para amostra

    return estimates, std_error