import warnings

import numpy as np
import pytest

from statstech.base import BaseEstimator, RegressorMixin
from statstech.linear_model.logistic_regression import LogisticRegression
from statstech.model_selection.search import GridSearchCV, _expand_grid
from statstech.model_selection.split import KFold
from statstech.model_selection.validation import cross_val_score
from statstech.utils.validation import NotFittedError, check_is_fitted


# Auxiliares
# ----------
class ShiftRegressor(RegressorMixin, BaseEstimator):
    """Modelo de teste: prevê o próprio X somado a um deslocamento."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return X[:, 0] + self.shift


class FailingRegressor(RegressorMixin, BaseEstimator):
    """Modelo de teste: falha no fit quando fail=True."""

    def __init__(self, fail=False):
        self.fail = fail

    def fit(self, X, y):
        if self.fail:
            raise ValueError("falha de propósito")
        self.fitted_ = True
        return self

    def predict(self, X):
        return X[:, 0]


def make_classification_data(n=60):
    """Devolve X (n, 2) e y (n,) com duas classes separáveis."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def make_data(n=20):
    """Devolve X (n, 1) e y (n,) com y igual a X."""
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()
    return X, y


# _expand_grid
# ------------
def test_expand_grid_cartesian_product():
    params = {"a": [1, 2], "b": [True, False]}
    combinations = _expand_grid(params)
    expected = [
        {"a": 1, "b": True},
        {"a": 1, "b": False},
        {"a": 2, "b": True},
        {"a": 2, "b": False},
    ]
    assert combinations == expected


# escolha do melhor
# -----------------
def test_best_params_picks_highest_r2():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)

    assert grid.best_params_ == {"shift": 0.0}
    assert grid.best_score_ == pytest.approx(1.0)


def test_lower_is_better_picks_smallest():
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, scoring="mse"
    ).fit(X, y)

    assert grid.best_params_ == {"shift": 0.0}
    assert grid.best_score_ == pytest.approx(0.0)


def test_best_score_matches_cv_results():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    scores = [
        r["mean_score"] for r in grid.cv_results_ if r["params"] == grid.best_params_
    ]
    assert len(scores) == 1
    assert scores[0] == pytest.approx(grid.best_score_)


# Resultados e entradas
# ---------------------
def test_cv_results_one_item_per_combination():
    """cv_results_ tem um item por combinação, na ordem da grade."""
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    assert len(grid.cv_results_) == 3
    for result in grid.cv_results_:
        assert set(result.keys()) == {"params", "mean_score", "std_score", "scores"}
    params_list = [result["params"] for result in grid.cv_results_]
    assert params_list == [{"shift": -5.0}, {"shift": 0.0}, {"shift": 5.0}]


def test_original_estimator_not_modified():
    X, y = make_data()
    model = ShiftRegressor()
    GridSearchCV(model, {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert model.shift == 0.0


# refit, predict e score
# ----------------------
def test_refit_true_trains_best_estimator():
    X, y = make_data()
    model = ShiftRegressor()
    grid = GridSearchCV(model, {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    check_is_fitted(grid.best_estimator_)
    assert grid.best_estimator_.shift == grid.best_params_["shift"]


def test_predict_and_score_use_best_estimator():
    """predict e score delegam ao melhor modelo (shift=0)."""
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4).fit(X, y)
    np.testing.assert_allclose(grid.predict(X), X[:, 0])
    assert grid.score(X, y) == pytest.approx(1.0)


def test_refit_false_predict_raises():
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, refit=False
    ).fit(X, y)
    with pytest.raises(NotFittedError):
        grid.predict(X)
    assert grid.best_params_ == {"shift": 0.0}


def test_predict_before_fit_raises():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4)
    with pytest.raises(NotFittedError):
        grid.predict(X)
    with pytest.raises(NotFittedError):
        grid.score(X, y)


# error_score
# -----------
@pytest.mark.parametrize("scoring", [None, "mse"])
def test_failing_combination_gets_nan_and_is_not_chosen(scoring):
    X, y = make_data()
    with pytest.warns(UserWarning, match="falhou"):
        grid = GridSearchCV(
            FailingRegressor(), {"fail": [True, False]}, cv=4, scoring=scoring
        ).fit(X, y)

    assert grid.best_params_ == {"fail": False}
    assert np.isnan(grid.cv_results_[0]["mean_score"])
    assert not np.isnan(grid.best_score_)


def test_error_score_number_is_used_as_score():
    X, y = make_data()
    with pytest.warns(UserWarning, match="falhou"):
        grid = GridSearchCV(
            FailingRegressor(), {"fail": [True, False]}, cv=4, error_score=-1.0
        ).fit(X, y)

    assert grid.cv_results_[0]["mean_score"] == -1.0
    assert grid.best_params_ == {"fail": False}


def test_error_score_raise_reraises_original_error():
    X, y = make_data()
    grid = GridSearchCV(
        FailingRegressor(), {"fail": [True, False]}, cv=4, error_score="raise"
    )
    with pytest.raises(ValueError, match="falha de propósito"):
        grid.fit(X, y)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_combinations_failing_raises():
    X, y = make_data()
    grid = GridSearchCV(FailingRegressor(), {"fail": [True]}, cv=4)
    with pytest.raises(ValueError, match="Todas"):
        grid.fit(X, y)


@pytest.mark.parametrize("error_score", ["abc", True, None, [1.0]])
def test_invalid_error_score_raises(error_score):
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [0.0]}, cv=4, error_score=error_score
    )
    with pytest.raises(ValueError, match="error_score"):
        grid.fit(X, y)


def test_no_warning_when_nothing_fails():
    X, y = make_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GridSearchCV(ShiftRegressor(), {"shift": [-5.0, 0.0]}, cv=4).fit(X, y)


# predict_proba e classes_
# ------------------------
def test_predict_proba_uses_best_estimator():
    X, y = make_classification_data()
    grid = GridSearchCV(LogisticRegression(), {"alpha": [0.1, 1.0]}, cv=3).fit(X, y)
    proba = grid.predict_proba(X)

    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba, grid.best_estimator_.predict_proba(X))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classes_copied_from_best_estimator():
    X, y = make_classification_data()
    labels = np.array(["a", "b"])[y]
    grid = GridSearchCV(LogisticRegression(), {"alpha": [1.0]}, cv=3).fit(X, labels)

    np.testing.assert_array_equal(grid.classes_, ["a", "b"])
    np.testing.assert_array_equal(grid.classes_, grid.best_estimator_.classes_)


def test_regressor_has_no_classes_nor_predict_proba():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [0.0]}, cv=4).fit(X, y)

    assert not hasattr(grid, "classes_")
    with pytest.raises(AttributeError):
        grid.predict_proba(X)


def test_predict_proba_before_fit_raises():
    X, _ = make_classification_data()
    grid = GridSearchCV(LogisticRegression(), {"alpha": [1.0]})
    with pytest.raises(NotFittedError):
        grid.predict_proba(X)


def test_predict_proba_with_refit_false_raises():
    X, y = make_classification_data()
    grid = GridSearchCV(LogisticRegression(), {"alpha": [1.0]}, cv=3, refit=False).fit(
        X, y
    )
    with pytest.raises(NotFittedError):
        grid.predict_proba(X)


def test_grid_search_inside_cross_val_score_with_log_loss():
    X, y = make_classification_data()
    grid = GridSearchCV(LogisticRegression(), {"alpha": [0.1, 1.0]}, cv=3)
    scores = cross_val_score(grid, X, y, cv=3, scoring="log_loss")

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores > 0)


# cv_results_ completo e best_index_
# ----------------------------------
def test_cv_results_has_all_keys():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [0.0, 5.0]}, cv=4).fit(X, y)
    for item in grid.cv_results_:
        assert set(item) == {"params", "mean_score", "std_score", "scores"}


def test_cv_results_one_score_per_fold():
    X, y = make_data()
    grid = GridSearchCV(ShiftRegressor(), {"shift": [0.0, 5.0]}, cv=4).fit(X, y)
    for item in grid.cv_results_:
        assert len(item["scores"]) == 4


def make_noisy_data(n=40):
    """Devolve X (n, 1) e y (n,) com ruído, para os folds terem notas diferentes."""
    rng = np.random.default_rng(0)
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = X[:, 0] + rng.normal(size=n)
    return X, y


def test_cv_results_mean_and_std_match_scores():
    X, y = make_noisy_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, scoring="mse"
    ).fit(X, y)
    for item in grid.cv_results_:
        assert item["std_score"] > 0
        assert item["mean_score"] == pytest.approx(np.mean(item["scores"]))
        assert item["std_score"] == pytest.approx(np.std(item["scores"], ddof=1))


def test_cv_results_scores_match_cross_val_score():
    X, y = make_data()
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [0.0, 5.0]}, cv=cv, scoring="mse"
    ).fit(X, y)
    for item in grid.cv_results_:
        model = ShiftRegressor().set_params(**item["params"])
        expected = cross_val_score(model, X, y, cv=cv, scoring="mse")
        np.testing.assert_allclose(item["scores"], expected)


@pytest.mark.parametrize("scoring", [None, "mse"])
def test_best_index_points_to_best(scoring):
    X, y = make_data()
    grid = GridSearchCV(
        ShiftRegressor(), {"shift": [-5.0, 0.0, 5.0]}, cv=4, scoring=scoring
    ).fit(X, y)
    best = grid.cv_results_[grid.best_index_]

    assert isinstance(grid.best_index_, int)
    assert best["params"] == grid.best_params_
    assert best["mean_score"] == grid.best_score_


def test_failed_combination_has_nan_std_and_no_scores():
    X, y = make_data()
    with pytest.warns(UserWarning, match="falhou"):
        grid = GridSearchCV(
            FailingRegressor(), {"fail": [True, False]}, cv=4, error_score=-1.0
        ).fit(X, y)
    failed = grid.cv_results_[0]

    assert np.isnan(failed["std_score"])
    assert failed["scores"] == []
    assert grid.best_index_ == 1
