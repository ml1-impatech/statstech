"""Busca em grade de hiperparâmetros com validação cruzada."""

import warnings
from itertools import product
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statstech.base import BaseEstimator, clone
from statstech.metrics.scorers import lower_is_better
from statstech.model_selection.validation import (
    EstimatorProtocol,
    ScoringCallable,
    SplitterProtocol,
    cross_val_score,
)
from statstech.utils.validation import check_is_fitted


def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Gera todas as combinações de valores de uma grade.

    Parameters
    ----------
    param_grid : dict
        Dicionário {nome_do_parâmetro: lista_de_valores}

    Returns
    -------
    combinations : list of dict
        Uma combinação por item, na forma {nome: valor}. A ordem
        segue o produto cartesiano das listas de valores.
    """
    return [
        dict(zip(param_grid.keys(), params)) for params in product(*param_grid.values())
    ]


class GridSearchCV(BaseEstimator):
    """Busca em grade de hiperparâmetros com validação cruzada.

    Testa todas as combinações de param_grid. Cada combinação é
    avaliada por cross_val_score e a de melhor média é escolhida.

    Parameters
    ----------
    estimator : objeto estimador
        Modelo com fit, predict e set_params. Ele não é modificado:
        a busca trabalha sempre com cópias.

    param_grid : dict
        Dicionário {nome_do_parâmetro: lista_de_valores}.

    cv : int ou splitter, default=5
        Número de folds ou objeto que gera as partições.

    scoring : str, callable ou None, default=None
        Nome de uma métrica, função (y_true, y_pred) -> float ou
        None para usar o score do próprio estimador. Quando for um
        nome em que menor é melhor (como "mse"), a busca escolhe a
        menor média; nos demais casos, a maior.
        Uma função passada diretamente é sempre maximizada.

    refit : bool, default=True
        Se True, treina best_estimator_ com a melhor combinação
        usando todos os dados.

    error_score : "raise" ou float, default=np.nan
        O que fazer quando uma combinação dá erro na validação cruzada.
        Se "raise", o erro interrompe a busca. Se for um número, a
        combinação recebe esse valor como nota, um aviso é emitido e a
        busca continua. Com o padrão np.nan, a combinação que falhou
        nunca é escolhida como a melhor.

    Attributes
    ----------
    cv_results_ : list of dict
        Um item por combinação, com as chaves:
        "params": a combinação de hiperparâmetros;
        "mean_score": média das notas dos folds;
        "std_score": desvio padrão amostral (ddof=1) das notas;
        "scores": lista com a nota de cada fold.

        Quando uma combinação falha, mean_score é o
        valor de error_score, std_score é NaN e scores é uma lista vazia.

    best_score_ : float
        Média do score da melhor combinação.

    best_params_ : dict
        Melhor combinação de hiperparâmetros.

    best_estimator_ : objeto estimador
        Modelo treinado com best_params_ nos dados completos.
        Só existe se refit=True.

    classes_ : ndarray de formato (n_classes,)
        Classes do melhor modelo. Só existe se refit=True e o
        estimador for um classificador.

    best_index_ : int
        Posição da melhor combinação em cv_results_.

    Warns
    -----
    UserWarning
        Quando uma combinação falha e error_score não é "raise".
    """

    def __init__(
        self,
        estimator: EstimatorProtocol,
        param_grid: dict[str, list[Any]],
        cv: int | SplitterProtocol = 5,
        scoring: str | ScoringCallable | None = None,
        refit: bool = True,
        error_score: float | Literal["raise"] = np.nan,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score

    def _clone_with(self, params: dict[str, Any]) -> Any:
        """Clona o estimador e aplica os hiperparâmetros dados."""
        model = clone(self.estimator)
        return model.set_params(**params)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Busca a melhor combinação de hiperparâmetros.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.
        y : array-like de formato (n_amostras,)
            Alvo.

        Returns
        -------
        self : GridSearchCV
            O próprio objeto, com os resultados da busca.

        Raises
        ------
        ValueError
            Se error_score não for "raise" nem um número, ou se todas
            as combinações falharem.

        Warns
        -----
        UserWarning
            Quando uma combinação falha e error_score não é "raise".
        """
        combinations = _expand_grid(self.param_grid)
        self.cv_results_: list[dict[str, Any]] = []
        means: list[float] = []

        is_raise = isinstance(self.error_score, str) and self.error_score == "raise"
        is_number = isinstance(self.error_score, (int, float)) and not isinstance(
            self.error_score, bool
        )
        if not (is_raise or is_number):
            raise ValueError("error_score deve ser 'raise' ou um número.")

        for params in combinations:
            model = self._clone_with(params)
            try:
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
                scores_mean = float(np.mean(scores))
                scores_std = float(np.std(scores, ddof=1))
                fold_scores = scores.tolist()
            except Exception as error:
                if is_raise:
                    raise
                warnings.warn(
                    f"A combinação {params} falhou e recebeu nota "
                    f"{self.error_score}: {type(error).__name__}: {error}",
                    UserWarning,
                    stacklevel=2,
                )
                scores_mean = float(self.error_score)
                scores_std = float("nan")
                fold_scores = []
            means.append(scores_mean)
            self.cv_results_.append(
                {
                    "params": params,
                    "mean_score": scores_mean,
                    "std_score": scores_std,
                    "scores": fold_scores,
                }
            )

        if np.all(np.isnan(means)):
            raise ValueError("Todas as combinações de parâmetros falharam.")
        if isinstance(self.scoring, str) and lower_is_better(self.scoring):
            best_idx = np.nanargmin(means)
        else:
            best_idx = np.nanargmax(means)
        self.best_index_: int = int(best_idx)
        self.best_score_: float = means[best_idx]
        self.best_params_: dict[str, Any] = combinations[best_idx]

        if self.refit:
            model = self._clone_with(self.best_params_)
            model.fit(X, y)
            self.best_estimator_: Any = model
            if hasattr(model, "classes_"):
                self.classes_: NDArray[Any] = model.classes_

        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Prevê com o melhor modelo encontrado.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.

        Returns
        -------
        y_pred : ndarray de formato (n_amostras,)
            Previsões de best_estimator_.

        Raises
        ------
        NotFittedError
            Se fit não foi chamado ou se refit=False.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Prevê as probabilidades de cada classe com o melhor modelo.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.

        Returns
        -------
        probabilities : ndarray de formato (n_amostras, n_classes)
            Probabilidades previstas por best_estimator_. As colunas
            seguem a ordem de classes_.

        Raises
        ------
        NotFittedError
            Se fit não foi chamado ou se refit=False.
        AttributeError
            Se o melhor modelo não tiver predict_proba (por exemplo,
            um regressor).
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Calcula o score do melhor modelo.

        Usa o método score do próprio estimador (acurácia ou R²),
        e não o parâmetro scoring da busca.

        Parameters
        ----------
        X : array-like de formato (n_amostras, n_features)
            Dados de entrada.
        y : array-like de formato (n_amostras,)
            Valores verdadeiros do alvo.

        Returns
        -------
        score : float
            Score de best_estimator_ em (X, y).

        Raises
        ------
        NotFittedError
            Se fit não foi chamado ou se refit=False.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.score(X, y)
