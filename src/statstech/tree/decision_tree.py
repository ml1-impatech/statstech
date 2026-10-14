"""Decision trees (CART) for regression and classification."""

import numpy as np

from dataclasses import dataclass

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class _Node:
    """
    Nó da árvore.
    
    Um nó interno tem ''feature'', ''threshold'', ''left'' e ''rignt''.
    Uma folha tem apenas''value''.
    """
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None  # só nas folhas

def _node_depth(node): # função auxiliar
    """
    Devolve a profundidade da subárvore que começa em ''node''.
    
    Parameters
    ----------
    node : _Node
        Raís da subárvore.

    Returns
    -------
    depth : int
        Número de divisões no caminho mais longo até uma folha.
        Uma folha tem profundidade 0
    """
    if node.feature is None:
        return 0
    return 1 + max(_node_depth(node.left), _node_depth(node.right))

class _BaseDecisionTree(BaseEstimator):
    """
    Classe-base das árvores de decisão
    
    Implementa a construção recursiva e a predição.
    As subclasses só definem o critério de impureza e o valor das folhas.

    Parameters
    ----------
    max_depth : int ou None, default=None
        Profundidade máxima da árvore.
        Se None, a árvore cresce até todos os nós ficarem puros ou pequenos demais para dividir.
    min_samples_split : int, default=2
        Número mínimo de amostras para tentar dividir um nó.

    Attributes
    ----------
    tree_ : _Node
        Nó raíz da árvore treinada.
    n_features_in_ : int
        Número de features vistas no "fit".
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    # ------------------------------------------------------------------
    # Validação (temporária: trocar por utils/validation quando existir)
    # ------------------------------------------------------------------
        
    def _check_params(self):
        """
        Valida os hiperparâmetros.

        Raises
        ------
        ValueError
            Se ''max_depth'' não for None nem um inteiro >= 1,
            ou se ''min_samples_split'' não for um inteiro >= 2.
        """
        if self.max_depth is not None:
            if (
                not isinstance(self.max_depth, (int, np.integer)) or
                self.max_depth < 1
            ):
                raise ValueError("max_depth deve ser None ou inteiro >= 1.")
        if (
            not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError("min_samples_split deve ser inteiro >= 2.")

    @staticmethod
    def _check_X(X): 
        """
        Converte X para um array 2D de floats.

        Parameters
        ----------
        X : array-like
            Dados de entrada.

        Returns
        -------
        X : ndarray de shape (n_amostras, n_features)
            cópia de X como float.

        Raises
        ------
        ValueError
            Se X não for 2D.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                "X must be 2D with shape (n_samples, n_features)."
                "For a single feature, use X.reshape(-1,1).")
        return X.astype(float)

    def _check_is_fitted(self):
        """
        Confere que o modelo foi treinado.
        
        Raises
        ------
        RuntimeError
            Se ''fit'' ainda não foi chamado.
        """
        if not hasattr(self, "tree_"):
            raise RuntimeError(
                f"Esta {type(self).__name__} ainda não passou pelo fit. Chame ''fit'' antes."
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Constrói a árvore de decisão a partir dos dados de treino.
        
        Parameters
        ----------
        X : array-like de shape (n_amostras, n_features)
            Dados de entrada (features).
        Y : array-like de shape (n_amostras,)
            Valores (regressão) ou rótulos (classificação) correspondentes a X.
            
        Returns
        -------
        self : objeto
            O próprio modelo treinado.
            
        Raises
        ------
        ValueError
            Se os hiperparâmetros forem inválidos.
            Se X não for 2D ou y não for 1D.
            Se não houver nenhuma amostra.
        """
        self._check_params()
        X = self._check_X(X)
        y = np.asarray(y)

        if y.dim != 1:
            raise ValueError("y deve ser 1D.")
        if len(y) != X.shape[0]:
            raise ValueError("X e y devem ter o mesmo número de linhas.")
        if len(y) == 0:
            raise ValueError("Não é possível aplicar o fit com zero amostras.")

        y = self._prepare_y(y)

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Prevê o alvo de cada amostra de X.

        Parameters
        ----------
        X : array-like de shape (n_amostras, n_features)
            Dados de entrada. Precisa ter o mesmo número de features do ''fit''.

        Returns
        -------
        y_pred : ndarray de shape (n_amostras,)
            Valores previstos (regressão) ou rótulos originais (classificação).
        
        Raises
        ------
        RuntimeError
            Se o modelo ainda não foi treinado.
        ValueError
            Se X não for 2D ou tiver número de features diferente do usado no ''fit''.
        """
        if not hasattr(self, "tree_"):
            raise RuntimeError(f"Esta {type(self).__name__} ainda não passou pelo fit. Chame ''fit'' antes.")
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} features, mas o modelo foi fitted com {self.n_features_in_}."
            )

        values = []
        for x in X: #desce pela árvore, x é linha da matriz
            node = self.tree_
            while node.feature is not None: # enquanto nó não é folha
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            values.append(node.value)

        return self._decode_predictions(np.array(values))

    def get_depth(self, y, depth):
        """
        Devolve a profundidade da árvore treinada.
        
        Útil para estudar overfitting: sem ''max_depth'' a profundidade cresce até a árvore decorar o treino.
        
        Returns
        -------
        depth : int
            Número de divisões no caminho mais longo da raíz até uma folha.
            Uma árvore com só uma folha tem profundidade 0.
            
        Raises
        ------
        RuntimeError
            Se o modelo ainda não foi treinado.
        """
        self._check_is_fitted()
        return _node_depth(self.tree_)

    # ------------------------------------------------------------------
    # Construção da árvore
    # ------------------------------------------------------------------
    
    def _is_leaf(self, y, depth):
        """
        Diz se um nó com alvo ``y`` e profundidade ``depth`` é folha.
 
        Parameters
        ----------
        y : ndarray de shape (n_amostras,)
            Alvo das amostras que chegaram ao nó.
        depth : int
            Profundidade do nó.
 
        Returns
        -------
        is_leaf : bool
            True se algum critério de parada for atendido.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            return True
 
        if len(y) < self.min_samples_split:
            return True
 
        if self._impurity(y) <= 1e-12:  # nó puro
            return True
 
        return False
    
    def _build(self, X, y, depth):
        """
        Constrói recursivamente a subárvore para as amostras (X, y).
        
        Parameters
        ----------
        X : ndarray de shape (n_amostras, n_features)
            Features das amostras que chegam ao nó.
        y : ndarray de shape (n_amostras,)
            Alvo preparado por ''_prepare_y''.
        depth : int
            Profundidade do nó atual.
            
        Returns
        -------
        node : _Node
            Folha ou nó interno com a melhor divisão encontrada.
        """
        n_samples = len(y)

        if self._is_leaf(y, depth):
            return _Node(value= self._leaf_value(y))
        
        # Busca da melhor divisão
        best_feature, best_threshold = None, None
        best_score = np.inf #infinito

        for j in range(X.shape[1]):
        # Limiares candidatos: pontos médios entre valores consecutivos distintos e ordenados da coluna j
            unique_values = np.unique(X[:, j]) 
            threshold = (
                unique_values[:, -1] 
                + unique_values[1:]
                )/2

            for t in threshold:
                mask = X[:, j] <= t #True = vai p esquerda
                y_left, y_right = y[mask], y[~mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue #pode acontecer por arredondamento de floats

                score = (
                    len(y_left)*self._impurity(y_left)
                    + len(y_right)*self._impurity(y_right)
                    )/n_samples

                if score < best_score:
                    best_score = score
                    best_feature = j
                    best_threshold = t
        
        if best_feature is None: #linhas idênticas
            return _Node(value=self._leaf_value(y))

        mask = X[:, best_feature] <= best_threshold
        node = _Node(feature=best_feature, threshold=best_threshold)
        node.left = self._build(X[mask], y[mask], depth+1)
        node.right = self._build(X[~mask], y[~mask], depth+1)
        return node
        
    def _prepare_y(self, y):
        pass

    def _decode_predictions(self, values):
        pass

    def _impurity(self, y):  # cada subclasse define
        pass

    def _leaf_value(self, y):  # cada subclasse define
        pass

class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def _impurity(self, y):
        def mean(y):
            result = 
            return result
        return mean((y- mean(y))**2)

    def _leaf_value(self, y):
        pass


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    def _impurity(self, y):
        Gini = 1 - sum(p_k**2) #p_k é a proporção da classe k em y
        counts = np.bincount(y)
        p = counts/len(y)

        # Nó puro é Gini = 0

        return 1 - np.sum(p**2)

    def _leaf_value(self, y):
        # empate: argmax pega o menor índice
        # fit já deixou os índices inteiros

        return np.argmax(np.bincount(y))
        