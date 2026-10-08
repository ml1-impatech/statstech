import numpy as np

from statstech.base import clone
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
