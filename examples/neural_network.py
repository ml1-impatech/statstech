"""Exemplo: rede neural para regressão.

Gera dados, treina uma MLPRegressor e mostra se a rede aprendeu
(o erro fica próximo da variância do ruído e a loss do fim do treino
é menor que a do começo).
"""

import numpy as np
from statstech.neural_network import MLPRegressor

rng = np.random.default_rng(0)

X = rng.normal(size=(200, 2))
y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * rng.normal(size=200)

model = MLPRegressor(
    hidden_layer_sizes=(16,), learning_rate=1e-2, max_iter=200, random_state=0
)
model.fit(X, y)
y_pred = model.predict(X)

# TODO: usar MSE de metrics
print("MSE:", np.mean((y_pred - y) ** 2))
print("Loss na primeira época:", model.loss_curve_[0])
print("Loss na última época:", model.loss_curve_[-1])
