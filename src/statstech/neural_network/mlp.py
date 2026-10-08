import numpy as np

from statstech.base import BaseEstimator, RegressorMixin


def relu(preactivation: np.ndarray) -> np.ndarray:
    """Função de ativação ReLU.
    
    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.
    
    Returns
    -------
    activation : np.ndarray
        Array com o resultado da ativação.
    """
    activation = np.maximum(0, preactivation)
    return activation


def relu_derivative(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a derivada da função de ativação ReLU.
    
    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.
    
    Returns
    -------
    np.ndarray
        Array com a derivada (1.0 para x > 0, 0.0 caso contrário).
    """
    return (preactivation > 0).astype(float)


def identity(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a função de ativação identidade.
    
    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.
    
    Returns
    -------
    np.ndarray
        Cópia do array de entrada mantendo os valores originais. 
    """
    return np.copy(preactivation)


def identity_derivative(preactivation: np.ndarray) -> np.ndarray:
    """Calcula a derivada da função de ativação identidade.
    
    Parameters
    ----------
    preactivation : np.ndarray
        Array com os valores de pré-ativação.
    
    Returns
    -------
    np.ndarray
        Array de uns com a mesma estrutura da entrada.
    """
    return np.ones_like(preactivation, dtype=float)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "identity": (identity, identity_derivative),
}


class MLPRegressor(BaseEstimator, RegressorMixin):
    """Multi-Layer Perceptron (MLP) Regressor.
    
    Implementação de uma rede neural para regressão
    
    Parameters
    ----------
    hidden_layer_sizes : tuple of itn, default=(32,)
        O número de neurônios em cada camada oculta.
    activation : {'relu', 'identity'}, default='relu'
        Função de ativação utilizada nas camadas ocultas.
    learning_rate : float, default=1e-3
        Taxa de aprendizado utilizada para a atualização dos pesos via 
        gradiente descendente.
    max_iter : int, default=200
        Número máximo de épocas de treinamento.
    batch_size : int, default=32
        Tamanho dos mini-batches para o treinamento.
    random_state : int, RandomState instance or None, default=None
        Semente para o gerador de números aleatórios.
    
    Attributes
    ----------
    (continuar)
        
    """

    def __init__(
        self,
        hidden_layer_sizes=(32,),
        activation="relu",
        learning_rate=1e-3,
        max_iter=200,
        batch_size=32,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _forward(self, X):
        raise NotImplementedError

    def _backward(self, X, y, activations):
        raise NotImplementedError
