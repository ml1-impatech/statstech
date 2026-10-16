import numpy as np

from statstech.base import BaseEstimator, RegressorMixin
from statstech.utils.validation import check_X_y, check_array, check_is_fitted


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

    Implementação de uma rede neural para regressão.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(32,)
        O número de neurônios em cada camada oculta.
    activation : {'relu', 'identity'}, default='relu'
        Função de ativação utilizada nas camadas ocultas.
    learning_rate : float, default=1e-3
        Taxa de aprendizado utilizada para a atualização dos pesos via
        gradiente descendente.
    max_iter : int, default=200
        Número de épocas de treinamento.
    batch_size : int, default=32
        Tamanho dos mini-batches para o treinamento.
    random_state : int or None, default=None
        Semente para o gerador de números aleatórios.

    Attributes
    ----------
    weights_ : list de ndarray
        Matrizes de pesos, uma por camada.
    biases_ : list de ndarray
        Vetores de viés, um por camada.
    loss_curve_ : list de float
        MSE no conjunto de treino ao fim de cada época.
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

    def _init_weights(self, n_features, rng):
        """Inicializa pesos e vieses de todas as camadas."""
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        weights = []
        biases = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2 / n_in)
            weights.append(rng.normal(0, scale, size=(n_in, n_out)))
            biases.append(np.zeros(n_out))

        self.weights_ = weights
        self.biases_ = biases

    def fit(self, X, y):
        """Treina a rede com gradiente descendente em mini-lotes.

        Parameters
        ----------
        X : array (n_amostras, n_features)
            Dados de treino.
        y : array (n_amostras,)
            Valores alvo.

        Returns
        -------
        self : MLPRegressor
            O modelo treinado.

        Raises
        ------
        ValueError
            Se `activation` não for uma das opções válidas,
            ou se X ou y forem inválidos.
        """
        X, y = check_X_y(X, y)
        X = X.astype(float)
        y = y.astype(float)
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Ativação inválida {self.activation!r}. "
                f"As opções válidas são: {list(ACTIVATIONS)}"
            )

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self._init_weights(n_features, rng)
        self.loss_curve_ = []

        for _ in range(self.max_iter):
            order = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                zs, activations = self._forward(X_batch)
                grad_W, grad_b = self._backward(y_batch, zs, activations)

                for l in range(len(self.weights_)):
                    self.weights_[l] -= self.learning_rate * grad_W[l]
                    self.biases_[l] -= self.learning_rate * grad_b[l]

            y_pred = self.predict(X)
            # TODO: mudar para utilizar metrics MSE(y_true, y_pred)
            self.loss_curve_.append(np.mean((y_pred - y) ** 2))
        return self

    def predict(self, X):
        """Prevê y para cada amostra de X.

        Parameters
        ----------
        X : array (n_amostras, n_features)
            Dados para os quais a previsão será feita.

        Returns
        -------
        y_pred : ndarray (n_amostras,)
            Valores previstos, um por amostra.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X não for 2D, tiver NaN ou infinitos, ou tiver
            número de features diferente do usado no treino.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        if X.shape[1] != self.weights_[0].shape[0]:
            raise ValueError(
                f"X tem {X.shape[1]} features, mas o modelo foi treinado "
                f"com {self.weights_[0].shape[0]}."
            )
        _, activations = self._forward(X)
        y_pred = activations[-1][:, 0]
        return y_pred

    def _forward(self, X):
        """Propaga X pela rede, camada por camada."""
        func, _ = ACTIVATIONS[self.activation]
        a = X
        zs = []
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            if i == len(self.weights_) - 1:
                a = identity(z)
            else:
                a = func(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def _backward(self, y, zs, activations):
        """Calcula os gradientes da loss por backpropagation."""
        n = len(y)
        _, der = ACTIVATIONS[self.activation]
        y_pred = activations[-1]
        y = y.reshape(-1, 1)
        delta = (2 / n) * (y_pred - y) * identity_derivative(zs[-1])
        grad_W = [None] * len(self.weights_)
        grad_b = [None] * len(self.biases_)
        for l in range(len(self.weights_) - 1, -1, -1):
            grad_W[l] = activations[l].T @ delta
            grad_b[l] = delta.sum(axis=0)
            if l != 0:
                delta = delta @ self.weights_[l].T * der(zs[l - 1])
        return grad_W, grad_b
