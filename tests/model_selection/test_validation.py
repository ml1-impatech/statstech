import numpy as np
import pytest

from statstech.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statstech.metrics.classification import log_loss
from statstech.metrics.regression import mean_squared_error
from statstech.model_selection.split import KFold, LeaveOneOut, StratifiedKFold
from statstech.model_selection.validation import cross_val_predict, cross_val_score
from statstech.utils.validation import NotFittedError, check_is_fitted


# Auxiliares: modelos, scorings e dados de teste
# ----------------------------------------------
class MeanRegressor(RegressorMixin, BaseEstimator):
    """Modelo de teste: prevê sempre a média do y de treino."""

    def fit(self, X, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class ConstantScoreRegressor(MeanRegressor):
    """Modelo de teste cujo score é sempre 7.0"""

    def score(self, X, y):
        return 7.0


class ZeroClassifier(ClassifierMixin, BaseEstimator):
    """Modelo de teste: prevê sempre a classe 0 (a menor).

    As probabilidades são fixas: 0.8 para a classe 0 e 0.2 para a outra.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        proba = np.full((len(X), 2), 0.2)
        proba[:, 0] = 0.8
        return proba


class FrequencyClassifier(ClassifierMixin, BaseEstimator):
    """Modelo de teste: a probabilidade de cada classe é a frequência
    dela no treino, igual para todas as amostras."""

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.freq_ = counts / counts.sum()
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[np.argmax(self.freq_)])

    def predict_proba(self, X):
        return np.tile(self.freq_, (len(X), 1))


class RepeatedFoldSplitter:
    """Divisor de teste que não é partição: devolve o mesmo fold duas vezes."""

    def split(self, X, y=None):
        idx = np.arange(len(X))
        for _ in range(2):
            yield idx[10:], idx[:10]


def constant_scoring(y_true, y_pred):
    return 10.0


def amp_pred_scoring(y_true, y_pred):
    return np.ptp(y_pred)


def make_data(n=20):
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n) * 2.0
    return X, y


def make_class_data(n=20):
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n) % 2
    return X, y


# cross_val_score: básico
# -----------------------
def test_one_score_per_fold():
    X, y = make_data()
    scores = cross_val_score(MeanRegressor(), X, y, cv=4)
    assert scores.shape == (4,)


def test_original_estimator_not_fitted():
    X, y = make_data()
    model = MeanRegressor()
    cross_val_score(model, X, y, cv=4)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


# cross_vall_score: scoring
# -------------------------
def test_default_scoring_uses_estimator_score():
    X, y = make_data()
    scores = cross_val_score(ConstantScoreRegressor(), X, y, cv=4)
    np.testing.assert_allclose(scores, 7.0)


def test_given_scoring_is_used():
    X, y = make_data()
    scores = cross_val_score(
        ConstantScoreRegressor(), X, y, cv=4, scoring=constant_scoring
    )
    np.testing.assert_allclose(scores, 10.0)


def test_str_scoring_is_used():
    X, y = make_class_data()
    scores = cross_val_score(ZeroClassifier(), X, y, cv=4, scoring="recall_macro")
    np.testing.assert_allclose(scores, 0.5)


def test_needs_proba_scoring_is_used():
    X, y = make_class_data()
    scores = cross_val_score(ZeroClassifier(), X, y, cv=4, scoring="log_loss")
    proba = np.tile([0.8, 0.2], (5, 1))
    fold_a = log_loss(np.array([0, 1, 0, 1, 0]), proba, [0, 1])
    fold_b = log_loss(np.array([1, 0, 1, 0, 1]), proba, [0, 1])
    expected = np.array([fold_a, fold_b, fold_a, fold_b])
    np.testing.assert_allclose(scores, expected)


def test_given_scoring_right_order():
    X, y = make_data()
    scores = cross_val_score(MeanRegressor(), X, y, cv=4, scoring=amp_pred_scoring)
    np.testing.assert_allclose(scores, 0.0)


def test_str_equals_function():
    X, y = make_data()
    scores_str_mse = cross_val_score(MeanRegressor(), X, y, cv=4, scoring="mse")
    scores_func_mse = cross_val_score(
        MeanRegressor(), X, y, cv=4, scoring=mean_squared_error
    )
    np.testing.assert_allclose(scores_str_mse, scores_func_mse)


# cross_val_score: parâmetro cv
# -----------------------------
def test_int_cv_kfold():
    X, y = make_data()
    scores_int = cross_val_score(MeanRegressor(), X, y, cv=4)
    scores_kfold = cross_val_score(MeanRegressor(), X, y, cv=KFold(n_splits=4))
    np.testing.assert_allclose(scores_int, scores_kfold)


def test_cv_object_is_used():
    X, y = make_data()
    scores_default = cross_val_score(MeanRegressor(), X, y, cv=KFold(n_splits=4))
    scores_shuffled = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    assert not np.allclose(scores_default, scores_shuffled)


def test_cv_same_seed_same_scores():
    X, y = make_data()
    scores_1 = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    scores_2 = cross_val_score(
        MeanRegressor(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    np.testing.assert_allclose(scores_1, scores_2)


def test_cv_accepts_loo():
    X, y = make_data()
    n = len(y)
    scores_loo = cross_val_score(ConstantScoreRegressor(), X, y, cv=LeaveOneOut())
    assert scores_loo.shape == (n,)
    np.testing.assert_allclose(scores_loo, 7.0)


# cross_val_score: entradas inválidas
# -----------------------------------
def test_int_scoring_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="scoring deve ser"):
        cross_val_score(MeanRegressor(), X, y, scoring=123)


def test_invalid_str_scoring_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="scoring desconhecido"):
        cross_val_score(MeanRegressor(), X, y, scoring="xyz")


def test_X_1d_raises():
    X, y = np.arange(20), np.arange(20)
    with pytest.raises(ValueError, match="precisa ser 2D"):
        cross_val_score(MeanRegressor(), X, y, cv=4)


def test_X_y_diff_lengths_raises():
    X, _ = make_data()
    y = np.arange(18)
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        cross_val_score(MeanRegressor(), X, y, cv=4)


def test_needs_proba_scoring_no_predict_proba_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="predict_proba"):
        cross_val_score(MeanRegressor(), X, y, scoring="log_loss")


# cross_val_score: StratifiedKFold
# --------------------------------
def test_cv_accepts_stratified_kfold():
    X, y = make_class_data()
    scores = cross_val_score(ZeroClassifier(), X, y, cv=StratifiedKFold(n_splits=5))
    assert scores.shape == (5,)


# cross_val_predict: predict
# --------------------------
def test_cvp_one_prediction_per_sample():
    X, y = make_data()
    pred = cross_val_predict(MeanRegressor(), X, y, cv=KFold(n_splits=4))

    assert pred.shape == (len(y),)
    for train_idx, test_idx in KFold(n_splits=4).split(X):
        np.testing.assert_allclose(pred[test_idx], y[train_idx].mean())


def test_cvp_loo_matches_manual():
    X, y = make_data(n=10)
    pred = cross_val_predict(MeanRegressor(), X, y, cv=LeaveOneOut())
    expected = (y.sum() - y) / (len(y) - 1)
    np.testing.assert_allclose(pred, expected)


def test_cvp_int_cv_uses_kfold():
    X, y = make_data()
    np.testing.assert_allclose(
        cross_val_predict(MeanRegressor(), X, y, cv=4),
        cross_val_predict(MeanRegressor(), X, y, cv=KFold(n_splits=4)),
    )


def test_cvp_keeps_float_predictions_with_int_y():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    pred = cross_val_predict(MeanRegressor(), X, y, cv=KFold(n_splits=5))

    assert np.issubdtype(pred.dtype, np.floating)
    assert pred[0] == pytest.approx(5.5)


def test_cvp_string_labels():
    X, y = make_class_data()
    labels = np.array(["a", "b"])[y]
    pred = cross_val_predict(ZeroClassifier(), X, labels, cv=4)
    assert pred.tolist() == ["a"] * len(y)


def test_cvp_original_estimator_not_fitted():
    X, y = make_data()
    model = MeanRegressor()
    cross_val_predict(model, X, y, cv=4)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


# cross_val_predict: predict_proba
# --------------------------------
def test_cvp_predict_proba_shape_and_sum():
    X, y = make_class_data()
    proba = cross_val_predict(FrequencyClassifier(), X, y, cv=4, method="predict_proba")
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_cvp_predict_proba_class_missing_from_fold():
    X = np.arange(7).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 2])
    proba = cross_val_predict(
        FrequencyClassifier(), X, y, cv=LeaveOneOut(), method="predict_proba"
    )

    assert proba.shape == (7, 3)
    # Sem a única amostra da classe 2, o modelo só conhece 0 e 1.
    np.testing.assert_allclose(proba[6], [0.5, 0.5, 0.0])
    # Sem uma amostra da classe 0, o treino tem 2, 3 e 1 de cada classe.
    np.testing.assert_allclose(proba[0], [2 / 6, 3 / 6, 1 / 6])


# cross_val_predict: entradas inválidas
# -------------------------------------
def test_cvp_invalid_method_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="method"):
        cross_val_predict(MeanRegressor(), X, y, method="abc")


def test_cvp_predict_proba_without_method_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="predict_proba"):
        cross_val_predict(MeanRegressor(), X, y, method="predict_proba")


def test_cvp_non_partition_cv_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="exatamente uma vez"):
        cross_val_predict(MeanRegressor(), X, y, cv=RepeatedFoldSplitter())
