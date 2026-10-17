"""Testes unitários para o classificador LogisticRegression."""

import numpy as np
import pytest

from statstech.linear_model.logistic_regression import LogisticRegression


def test_fit_multiclasse():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]

    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    assert np.array_equal(model.classes_, np.array([0, 1, 2]))
    assert model.coef_.shape == (2, 3)
    assert model.intercept_.shape == (3,)
    assert len(model.loss_history_) > 0


def test_loss_decreases():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]

    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    assert model.loss_history_[-1] < model.loss_history_[0]


def test_fit_respect_max_iter():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=5, tol=0)
    model.fit(X, y)

    assert len(model.loss_history_) == 5


def test_fit_convergence_criterion():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100, tol=100)
    model.fit(X, y)

    assert len(model.loss_history_) == 2


def test_gradient_finite_difference():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    Y = np.zeros((len(y), len(model.classes_)))

    for i in range(len(y)):
        Y[i, model.classes_ == y[i]] = 1

    coef = np.zeros(model.coef_.shape)
    intercept = np.zeros(model.intercept_.shape)

    _, gradient, _ = model._loss_and_gradient(X, Y, coef, intercept)

    h = 1e-5

    numerical_gradient = np.zeros(coef.shape)

    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            coef_incr = coef.copy()
            coef_decr = coef.copy()

            coef_incr[i, j] += h
            coef_decr[i, j] -= h

            loss_incr, _, _ = model._loss_and_gradient(
                X, Y, coef_incr, intercept
            )
            loss_decr, _, _ = model._loss_and_gradient(
                X, Y, coef_decr, intercept
            )

            numerical_gradient[i, j] = (loss_incr - loss_decr) / (2 * h)

    differ = np.abs(gradient - numerical_gradient)

    for i in range(differ.shape[0]):
        for j in range(differ.shape[1]):
            assert differ[i, j] < 1e-5


def test_predict_proba_interval():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    for i in range(len(prob)):
        for j in range(len(model.classes_)):
            assert prob[i, j] >= 0
            assert prob[i, j] <= 1


def test_predict_proba_sum_one():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    for i in range(len(prob)):
        total = 0

        for j in range(len(model.classes_)):
            total += prob[i, j]

        assert abs(total - 1) < 1e-10


def test_predict_proba_format():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    prob = model.predict_proba(X)

    assert prob.shape == (len(X), len(model.classes_))


def test_predict_text_labels():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array(["não", "não", "sim", "sim", "talvez", "talvez"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    for i in range(len(predictions)):
        assert predictions[i] in model.classes_


def test_predict_format():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [8.0, 9.0],
        [9.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)


def test_predict_high_accuracy_separated_data():
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [9.0, 9.0],
        [9.0, 10.0],
        [10.0, 9.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    correct = 0

    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct += 1

    accuracy = correct / len(y)

    assert accuracy >= 0.9


def test_predict_choose_highest_probability():
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [9.0, 9.0],
        [9.0, 10.0],
        [10.0, 9.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    for i in range(len(X)):
        index = 0

        for j in range(1, len(model.classes_)):
            if probabilities[i, j] > probabilities[i, index]:
                index = j

        assert predictions[i] == model.classes_[index]


def test_fit_intercept_false():
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [9.0, 9.0],
        [9.0, 10.0],
        [10.0, 9.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = LogisticRegression(max_iter=1000, fit_intercept=False)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.any(model.coef_ != 0)


def test_regularization_l2():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [5.0, 6.0],
        [8.0, 9.0],
        [9.0, 8.0],
        [9.0, 10.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model_without_l2 = LogisticRegression(max_iter=1000, lambda_=0)
    model_with_l2 = LogisticRegression(max_iter=1000, lambda_=1)

    model_without_l2.fit(X, y)
    model_with_l2.fit(X, y)

    diff = 0

    for i in range(model_without_l2.coef_.shape[0]):
        for j in range(model_without_l2.coef_.shape[1]):
            if model_without_l2.coef_[i, j] != model_with_l2.coef_[i, j]:
                diff += 1

    assert diff > 0


def test_learning_rate_invalid():
    model = LogisticRegression(learning_rate=0)

    with pytest.raises(
        ValueError,
        match="learning_rate deve ser positivo e finito"
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_max_iter_invalid():
    model = LogisticRegression(max_iter=0)

    with pytest.raises(
        ValueError,
        match="max_iter deve ser um inteiro positivo"
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_tol_invalid():
    model = LogisticRegression(tol=-1)

    with pytest.raises(
        ValueError,
        match="tol deve ser não negativo e finito"
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_fit_intercept_invalid():
    model = LogisticRegression(fit_intercept=1)

    with pytest.raises(
        ValueError,
        match="fit_intercept deve ser booleano"
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_lambda_invalid():
    model = LogisticRegression(lambda_=-1)

    with pytest.raises(
        ValueError,
        match="lambda_ deve ser não negativo e finito"
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_loss_and_gradient_different_samples():
    model = LogisticRegression()
    X = np.zeros((3, 2))
    Y = np.zeros((2, 2))
    coef = np.zeros((2, 2))
    intercept = np.zeros(2)

    with pytest.raises(
        ValueError,
        match="X e Y devem ter o mesmo número de observações",
    ):
        model._loss_and_gradient(X, Y, coef, intercept)


def test_fit_X_not_2d():
    model = LogisticRegression()
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    with pytest.raises(
        ValueError,
        match="Esse array é 1D, precisa ser 2D",
    ):
        model.fit(X, y)


def test_fit_y_not_1d():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])

    with pytest.raises(
        ValueError,
        match=r"y tem formato \(n, 1\), mas precisa ser 1D",
    ):
        model.fit(X, y)


def test_fit_different_samples():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="X e y têm tamanhos diferentes",
    ):
        model.fit(X, y)


def test_fit_empty_X():
    model = LogisticRegression()
    X = np.empty((0, 2))
    y = np.array([])

    with pytest.raises(
        ValueError,
        match="O array está vazio",
    ):
        model.fit(X, y)


def test_fit_y_nan():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, np.nan, 1])

    with pytest.raises(
        ValueError,
        match="y contém NaN ou valores infinitos",
    ):
        model.fit(X, y)


def test_fit_no_features():
    model = LogisticRegression()
    X = np.empty((3, 0))
    y = np.array([0, 1, 0])

    with pytest.raises(
        ValueError,
        match="O array está vazio",
    ):
        model.fit(X, y)


def test_fit_X_non_finite():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, np.inf], [5, 6]])
    y = np.array([0, 1, 0])

    with pytest.raises(
        ValueError,
        match="X contém NaN ou valores infinitos",
    ):
        model.fit(X, y)


def test_fit_single_class():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 1, 1])

    with pytest.raises(
        ValueError,
        match="y deve ter pelo menos duas classes",
    ):
        model.fit(X, y)


def test_predict_proba_model_not_fitted():
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(
        ValueError,
        match="ainda não foi treinado",
    ):
        model.predict_proba(X)


def test_predict_proba_X_not_2d():
    model = LogisticRegression()
    model.fit(
        np.array([[1, 2], [3, 4]]),
        np.array([0, 1])
    )
    X = np.array([1, 2])

    with pytest.raises(
        ValueError,
        match="Esse array é 1D, precisa ser 2D",
    ):
        model.predict_proba(X)


def test_predict_proba_different_features():
    model = LogisticRegression()
    model.fit(
        np.array([[1, 2], [3, 4]]),
        np.array([0, 1])
    )
    X = np.array([[1, 2, 3]])

    with pytest.raises(
        ValueError,
        match="X deve ter o mesmo número de características usado no fit",
    ):
        model.predict_proba(X)


def test_predict_proba_empty_X():
    model = LogisticRegression()
    model.fit(
        np.array([[1, 2], [3, 4]]),
        np.array([0, 1])
    )
    X = np.empty((0, 2))

    with pytest.raises(
        ValueError,
        match="O array está vazio",
    ):
        model.predict_proba(X)


def test_predict_proba_X_non_finite():
    model = LogisticRegression()
    model.fit(
        np.array([[1, 2], [3, 4]]),
        np.array([0, 1])
    )
    X = np.array([[1, np.inf]])

    with pytest.raises(
        ValueError,
        match="X contém NaN ou valores infinitos",
    ):
        model.predict_proba(X)


def test_learning_rate_tipo_invalido():
    model = LogisticRegression(learning_rate="abc")

    with pytest.raises(
        ValueError,
        match="learning_rate deve ser positivo e finito",
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_max_iter_booleano():
    model = LogisticRegression(max_iter=True)

    with pytest.raises(
        ValueError,
        match="max_iter deve ser um inteiro positivo",
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_tol_tipo_invalido():
    model = LogisticRegression(tol="abc")

    with pytest.raises(
        ValueError,
        match="tol deve ser não negativo e finito",
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))


def test_lambda_tipo_invalido():
    model = LogisticRegression(lambda_="abc")

    with pytest.raises(
        ValueError,
        match="lambda_ deve ser não negativo e finito",
    ):
        model.fit(np.array([[1], [2]]), np.array([0, 1]))
