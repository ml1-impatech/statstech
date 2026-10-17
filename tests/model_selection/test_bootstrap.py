
import numpy as np
import pytest 

from statstech.model_selection.bootstrap import bootstrap


# Reprodutibilidade
# -----------------
def test_same_seed_same_result():
    data = np.arange(50)
    statistic = np.mean
    est1, se1 = bootstrap(data, statistic, random_state=47)
    est2, se2 = bootstrap(data, statistic, random_state=47)
    np.testing.assert_allclose(est1, est2)
    np.testing.assert_allclose(se1, se2)


def test_diff_seed_diff_result():
    data = np.arange(50)
    statistic = np.mean
    est1, se1 = bootstrap(data, statistic, random_state=47)
    est2, se2 = bootstrap(data, statistic, random_state=74)
    assert not np.allclose(est1, est2)
    assert not np.allclose(se1, se2)


# Formato de saída 
# ----------------
def test_output_shape_scalar_stat():
    data = np.arange(100)
    statistic = np.mean
    est, se = bootstrap(data, statistic, n_bootstrap=200)
    assert est.shape == (200,)
    assert np.ndim(se) == 0


def test_output_shape_vector_stat():
    data = np.arange(100)
    def statistic(x):
        return np.array([np.mean(x), np.std(x)])
    est, se = bootstrap(data, statistic, n_bootstrap=200)
    assert est.shape == (200, 2)
    assert se.shape == (2,)


# Correção numérica
# -----------------
def test_numerical_correctness():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5, scale=2, size=200)
    statistic = np.mean
    est, se = bootstrap(data, statistic, n_bootstrap=2000, random_state=0)
    np.testing.assert_allclose(se, np.std(data)/np.sqrt(200), atol=0.05)
    np.testing.assert_allclose(est.mean(), data.mean(), atol=0.02)


def test_tuple_pairs_aligned():
    X = np.arange(50, dtype = float)
    y = 2 * X + 1
    data = (X, y)
    def statistic(X_boot, y_boot):
        return np.mean(y_boot - 2 * X_boot)
    est, se = bootstrap(data, statistic, random_state=0)
    np.testing.assert_allclose(est, 1.0)
    assert est.shape == (1000,)


# Casos degenerados
# -----------------
def test_constant_data():
    rng = np.random.default_rng(0)
    data = np.full(30, 5.0)
    statistic = np.mean
    est, se = bootstrap(data, statistic, n_bootstrap=2000, random_state=rng)
    np.testing.assert_allclose(est, 5.0)
    np.testing.assert_allclose(se, 0, atol=1e-12)


def test_single_sample():
    data = np.array([7.0])
    est, se = bootstrap(data, statistic = np.mean, random_state=0)
    np.testing.assert_allclose(est, 7.0)
    np.testing.assert_allclose(se, 0, atol=1e-12)


# Entradas inválidas
# ------------------
def test_n_bootstrap_less_than_2_raises():
    data = np.arange(10)
    statistic = np.mean
    with pytest.raises(ValueError, match="n_bootstrap deve ser pelo menos 2."):
        bootstrap(data, statistic, n_bootstrap=1)


def test_empty_tuple_raises():
    data = ()
    with pytest.raises(ValueError, match="Nenhum dado fornecido"):
        bootstrap(data, np.mean, random_state=0)


def test_empty_data_raises():
    data = np.array([])
    with pytest.raises(ValueError, match="O tamanho do dado deve ser maior que 0"):
        bootstrap(data, np.mean, random_state=0)


def test_diff_lenghts_raises():
    X = np.arange(10)
    y = np.arange(8)
    data = (X, y)
    with pytest.raises(ValueError, match="Todos os arrays devem ter o mesmo tamanho"):
        bootstrap(data, np.mean, random_state=0)
