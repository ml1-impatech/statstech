import numpy as np
import pytest

from statstech.base import clone
from statstech.utils.validation import NotFittedError
from statstech.neural_network.mlp import (
    MLPRegressor,
    identity,
    identity_derivative,
    relu,
    relu_derivative,
)


def test_get_params_and_clone():
    model = MLPRegressor(hidden_layer_sizes=(8, 4), learning_rate=0.01)
    new = clone(model)
    assert new is not model
    assert new.get_params() == model.get_params()


def test_relu():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 3.0])
    np.testing.assert_allclose(relu(z), expected)


def test_relu_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(relu_derivative(z), expected)


def test_identity():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(identity(z), expected)


def test_identity_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(identity_derivative(z), expected)


def test_activations_keep_shape():
    z = np.array([[-1.0, 2.0], [3.0, -4.0]])
    for func in [relu, relu_derivative, identity, identity_derivative]:
        assert func(z).shape == z.shape


def test_init_weights_shapes():
    model = MLPRegressor(hidden_layer_sizes=(32, 16))
    model._init_weights(5, np.random.default_rng(0))
    assert [w.shape for w in model.weights_] == [(5, 32), (32, 16), (16, 1)]
    assert [b.shape for b in model.biases_] == [(32,), (16,), (1,)]


def test_init_weights_biases():
    model = MLPRegressor(hidden_layer_sizes=(32, 16))
    model._init_weights(5, np.random.default_rng(0))
    for b in model.biases_:
        np.testing.assert_allclose(b, 0)


def test_forward_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([-10.0])]
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    zs, activations = model._forward(X)
    np.testing.assert_allclose(activations[-1], [[-5.0], [-6.0]])


def test_predict_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([1.0])]
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, [6.0, 5.0])
    assert y_pred.shape == (2,)


def test_predict_with_hidden_layer():
    model = MLPRegressor(hidden_layer_sizes=(3,))
    model.weights_ = [
        np.array([[1.0, -1.0, 0.0], [2.0, 1.0, 1.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ]
    model.biases_ = [np.array([0.0, -3.0, 0.0]), np.array([0.5])]
    X = np.array([[1.0, 2.0]])
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, [7.5])


def test_backward_linear():
    model = MLPRegressor(hidden_layer_sizes=())
    model.weights_ = [np.array([[2.0], [3.0]])]
    model.biases_ = [np.array([1.0])]
    X = np.array([[1.0, 1.0]])
    y = np.array([4.0])
    zs, activations = model._forward(X)
    grad_W, grad_b = model._backward(y, zs, activations)
    np.testing.assert_allclose(grad_W[0], [[4.0], [4.0]])
    np.testing.assert_allclose(grad_b[0], [4.0])


def test_backward_with_hidden_layer():
    model = MLPRegressor(hidden_layer_sizes=(3,))
    model.weights_ = [
        np.array([[1.0, -1.0, 0.0], [2.0, 1.0, 1.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ]
    model.biases_ = [np.array([0.0, -3.0, 0.0]), np.array([0.5])]
    X = np.array([[1.0, 2.0]])
    y = np.array([5.5])
    zs, activations = model._forward(X)
    grad_W, grad_b = model._backward(y, zs, activations)
    np.testing.assert_allclose(grad_W[0], [[4.0, 0.0, 4.0], [8.0, 0.0, 8.0]])
    np.testing.assert_allclose(grad_b[0], [4.0, 0.0, 4.0])
    np.testing.assert_allclose(grad_W[1], [[20.0], [0.0], [8.0]])
    np.testing.assert_allclose(grad_b[1], [4.0])


def test_fit_returns_self():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1, random_state=0)
    assert model.fit(X, y) is model


def test_fit_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        max_iter=200,
        learning_rate=1e-2,
        random_state=0,
    )
    model.fit(X, y)
    assert model.loss_curve_[-1] < model.loss_curve_[0]


def test_fit_same_seed_same_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model_a = MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=0)
    model_b = MLPRegressor(hidden_layer_sizes=(4,), max_iter=5, random_state=0)
    model_a.fit(X, y)
    model_b.fit(X, y)
    for wa, wb in zip(model_a.weights_, model_b.weights_):
        np.testing.assert_allclose(wa, wb)
    for ba, bb in zip(model_a.biases_, model_b.biases_):
        np.testing.assert_allclose(ba, bb)


def test_predict_before_fit_raises():
    model = MLPRegressor()
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 2)))


def test_predict_wrong_n_features_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=1, random_state=0)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 3)))
