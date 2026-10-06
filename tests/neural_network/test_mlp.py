from statstech.base import clone
from statstech.neural_network.mlp import MLPRegressor

def teste_get_params_and_clone():
    model = MLPRegressor(hidden_layer_sizes=(8, 4), learning_rate=0.01)
    new = clone(model)
    assert new is not model
    assert new.get_params() == model.get_params()
    