import numpy as np
from statstech.base import BaseEstimator, TransformerMixin
from statstech.utils.validation import check_array, check_is_fitted


class PCA(BaseEstimator, TransformerMixin):
    """Análise de componentes principais (PCA).

    Encontra as direções em que os dados mais variam a partir da SVD
    do X centralizado. As linhas de Vt são as componentes, e a variância
    de cada uma é s_i² / (n - 1).
    O PCA só centraliza os dados, não padroniza. Se as features
    estiverem em escalas muito diferentes, recomenda-se usar o
    StandardScaler antes.

    Parameters
    ----------
    n_components : int, float ou None, default=None
        Número de componentes para manter. Se None, todas, ou seja,
        o mínimo entre n_amostras e n_features. Se int entre 1 e o mínimo
        entre n_amostras e n_features, o próprio n_components. Se float
        entre 0 e 1, o menor número de componentes em que a variância
        acumulada atinge essa fração.

    Attributes
    ----------
    components_ : ndarray de formato (n_components_, n_features)
        Direções principais, com linhas ortonormais. Convenção de
        sinal: a maior entrada em módulo de cada componente é positiva.
    explained_variance_ : ndarray de formato (n_components_,)
        Variância dos dados ao longo de cada componente.
    explained_variance_ratio_ : ndarray de formato (n_components_,)
        Fração da variância total explicada por cada componente.
    mean_ : ndarray de formato (n_features,)
        Média de cada feature, usada para centralizar dados novos.
    n_components_ : int
        Número de componentes mantidas.
    n_features_in_ : int
        Número de features vistas no fit.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Aprende as componentes principais de X.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino.
        y : None
            Ignorado. Existe só para manter o padrão.

        Returns
        -------
        self : PCA
            O próprio modelo, treinado.

        Raises
        ------
        ValueError
            Se X tiver menos de 2 amostras, variância zero, ou se
            n_components for inválido.
        """
        X = check_array(X)
        n = X.shape[0]
        if n < 2:
            raise ValueError("PCA precisa de pelo menos 2 amostras")
        max_components = min(X.shape)
        if self.n_components is None:
            k = max_components
        elif (
            isinstance(self.n_components, (int, np.integer))
            and not isinstance(self.n_components, bool)
            and 1 <= self.n_components <= max_components
        ):
            k = self.n_components
        elif (
            isinstance(self.n_components, (float, np.floating))
            and 0 < self.n_components < 1
        ):
            k = None
        else:
            raise ValueError(
                f"n_components deve ser um inteiro entre 1 e "
                f"{max_components} ou um float entre 0 e 1, "
                f"mas é {self.n_components!r}"
            )
        self.mean_ = np.mean(X, axis=0)
        self.n_features_in_ = X.shape[1]
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        idx = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), idx])
        Vt = Vt * signs[:, np.newaxis]
        variances = S**2 / (n - 1)
        variances_total = np.sum(variances)
        if variances_total == 0:
            raise ValueError("X tem variância zero: todas as linhas são iguais")
        ratios = variances / variances_total
        if k is None:
            k = int(np.searchsorted(np.cumsum(ratios), self.n_components))
            k = min(k + 1, max_components)
        self.components_ = Vt[:k]
        self.n_components_ = k
        self.explained_variance_ = variances[:k]
        self.explained_variance_ratio_ = ratios[:k]
        return self

    def transform(self, X):
        """Pega dados de treino ou novos e escreve cada ponto nas
        coordenadas das componentes que o fit aprendeu.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de treino ou novos.

        Returns
        -------
        Z : ndarray de formato (n_amostras, n_components_)
            Coordenadas de cada ponto nas componentes principais.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        ValueError
            Se X tiver um número de features diferente do fit.
        """
        attributes = ["mean_", "components_", "n_features_in_"]
        check_is_fitted(self, attributes)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features mas deveria ter {self.n_features_in_}"
            )
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z):
        """Recebe as coordenadas Z e reconstrói os pontos no espaço original
        Com todas as componentes, a reconstrução é exata. Com menos,
        é a melhor aproximação de posto n_components_.

        Parameters
        ----------
        Z : array-like de formato (n_amostras, n_components_)
            Coordenadas nas componentes principais.

        Returns
        -------
        X : ndarray de formato (n_amostras, n_features)
            Pontos reconstruídos no espaço original.

        Raises
        ------
        NotFittedError
            Se o modelo ainda não tiver sido treinado.
        ValueError
            Se Z tiver um número de colunas diferente de n_components_.
        """
        attributes = ["mean_", "n_components_", "components_"]
        check_is_fitted(self, attributes)
        Z = check_array(Z)
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Z tem {Z.shape[1]} componentes, mas deveria ter {self.n_components_}"
            )
        Zd = Z @ self.components_
        Zdc = Zd + self.mean_
        return Zdc
