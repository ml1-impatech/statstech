import numpy as np
import pytest

from statstech.model_selection.split import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    train_test_split,
)


def make_data(n=10):
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n)
    return X, y


# train_test_split: tamanho dos conjuntos
# ---------------------------------------
def test_float_test_size_rounds_up():
    X, y = make_data()
    X_train, X_test, _, _ = train_test_split(X, y, random_state=47)
    assert len(X_test) == 3
    assert len(X_train) == 7


def test_int_test_size_absolute_count():
    X, y = make_data()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=4, random_state=47)
    assert len(X_test) == 4
    assert len(X_train) == 6


# train_test_split: test_size inválido
# ------------------------------------
def test_float_test_size_one_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="float entre 0 e 1"):
        train_test_split(X, y, test_size=1.0)


def test_float_test_size_zero_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="float entre 0 e 1"):
        train_test_split(X, y, test_size=0.0)


def test_int_test_size_equal_to_n_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="inteiro entre 0 e n"):
        train_test_split(X, y, test_size=10)


def test_int_test_size_zero_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="inteiro entre 0 e n"):
        train_test_split(X, y, test_size=0)


def test_str_test_size_raises():
    X, y = make_data()
    with pytest.raises(TypeError, match="float ou um inteiro"):
        train_test_split(X, y, test_size="0.5")


# train_test_split: dimensão das entradas
# ---------------------------------------
def test_1d_raises():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="precisa ser 2D"):
        train_test_split(X, y)


def test_X_y_different_lengths_raises():
    X, _ = make_data(10)
    y = np.arange(8)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        train_test_split(X, y)


# train_test_split: integridade dos dados
# ---------------------------------------
def test_train_and_test_cover_all_samples():
    X, y = make_data()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=47)
    X_all = np.concatenate([X_train, X_test])
    np.testing.assert_array_equal(np.sort(X_all, axis=0), X)


def test_X_and_y_stay_aligned():
    X, _ = make_data()
    y = X.ravel() * 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=47
    )
    np.testing.assert_array_equal(X_train.ravel() * 10, y_train)
    np.testing.assert_array_equal(X_test.ravel() * 10, y_test)


# train_test_split: reprodutibilidade
# -----------------------------------
def test_same_seed_same_split():
    X, y = make_data(100)
    split1 = train_test_split(X, y, test_size=0.3, random_state=47)
    split2 = train_test_split(X, y, test_size=0.3, random_state=47)
    for part1, part2 in zip(split1, split2):
        np.testing.assert_array_equal(part1, part2)


def test_diff_seeds_diff_splits():
    X, y = make_data(100)
    split1 = train_test_split(X, y, test_size=0.3, random_state=47)
    split2 = train_test_split(X, y, test_size=0.3, random_state=74)
    for part1, part2 in zip(split1, split2):
        assert not np.array_equal(part1, part2)


# train_test_split: shuffle=False
# -------------------------------
def test_shuffle_false_keeps_original_order():
    X, y = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    np.testing.assert_array_equal(X_train, np.arange(7).reshape(-1, 1))
    np.testing.assert_array_equal(X_test, np.arange(7, 10).reshape(-1, 1))
    np.testing.assert_array_equal(y_train, np.arange(7))
    np.testing.assert_array_equal(y_test, np.arange(7, 10))


# train_test_split: stratify
# --------------------------
def test_stratify_keeps_class_proportions():
    X, _ = make_data(20)
    y = np.array([0] * 15 + [1] * 5)
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    np.testing.assert_array_equal(np.bincount(y_train), [12, 4])
    np.testing.assert_array_equal(np.bincount(y_test), [3, 1])


def test_stratify_wrong_length_raises():
    X, y = make_data(10)
    stratify = np.array([0, 1] * 4)  # 8 rótulos para 10 amostras
    with pytest.raises(ValueError, match="X e stratify"):
        train_test_split(X, y, stratify=stratify)


def test_stratify_without_shuffle_raises():
    X, y = make_data(10)
    stratify = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="exige shuffle=True"):
        train_test_split(X, y, stratify=stratify, shuffle=False)


def test_stratify_class_with_one_sample_raises():
    X, y = make_data(10)
    stratify = np.array([0] * 9 + [1])
    with pytest.raises(ValueError, match="pelo menos 2"):
        train_test_split(X, y, stratify=stratify)


# KFold
# -----
def test_kfold_each_sample_tested_once():
    X = np.arange(13)
    splits = list(KFold(n_splits=5).split(X))
    all_test = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(np.sort(all_test), X)


def test_kfold_train_test_complementary():
    X = np.arange(13)
    for train_idx, test_idx in KFold(n_splits=5).split(X):
        assert np.intersect1d(train_idx, test_idx).size == 0
        all_idx = np.concatenate([train_idx, test_idx])
        np.testing.assert_array_equal(np.sort(all_idx), X)


def test_kfold_number_and_size_of_folds():
    X = np.arange(13)
    splits = list(KFold(n_splits=5).split(X))
    assert len(splits) == 5
    test_sizes = [len(test) for _, test in splits]
    assert test_sizes == [3, 3, 3, 2, 2]


def test_kfold_n_splits_less_than_2_raises():
    with pytest.raises(ValueError, match="pelo menos 2"):
        KFold(n_splits=1)


def test_kfold_n_splits_greater_than_n_raises():
    X = np.arange(13)
    with pytest.raises(ValueError, match="maior que n"):
        list(KFold(n_splits=len(X) + 1).split(X))


def test_kfold_same_seed_same_folds():
    X = np.arange(13)
    folds_1 = list(KFold(shuffle=True, random_state=0).split(X))
    folds_2 = list(KFold(shuffle=True, random_state=0).split(X))
    for (train_1, test_1), (train_2, test_2) in zip(folds_1, folds_2):
        np.testing.assert_array_equal(train_1, train_2)
        np.testing.assert_array_equal(test_1, test_2)


def test_kfold_shuffle_changes_folds():
    X = np.arange(13)
    folds_1 = list(KFold(shuffle=True, random_state=0).split(X))
    folds_2 = list(KFold(random_state=0).split(X))
    assert not np.array_equal(folds_1[0][1], folds_2[0][1])


# LeaveOneOut
# -----------
def test_loo_number_and_size_of_splits():
    X = np.arange(13)
    splits = list(LeaveOneOut().split(X))
    assert len(splits) == len(X)
    for train_idx, test_idx in splits:
        assert len(train_idx) == len(X) - 1
        assert len(test_idx) == 1


def test_loo_each_sample_tested_once():
    X = np.arange(13)
    splits = list(LeaveOneOut().split(X))
    all_test = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(np.sort(all_test), X)


def test_loo_samples_less_than_2_raises():
    X = np.arange(1)
    with pytest.raises(ValueError, match="pelo menos 2"):
        list(LeaveOneOut().split(X))


# KFold e LeaveOneOut com y
# -------------------------
def test_kfold_ignores_y():
    X, y = make_data()
    with_y = list(KFold(n_splits=5).split(X, y))
    without_y = list(KFold(n_splits=5).split(X))
    for (tr_a, te_a), (tr_b, te_b) in zip(with_y, without_y):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_loo_ignores_y():
    X, y = make_data()
    with_y = list(LeaveOneOut().split(X, y))
    without_y = list(LeaveOneOut().split(X))
    for (tr_a, te_a), (tr_b, te_b) in zip(with_y, without_y):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


# StratifiedKFold
# ---------------
def make_imbalanced_data():
    """50 amostras: 40 da classe 0 e 10 da classe 1."""
    X = np.arange(50).reshape(-1, 1)
    y = np.array([0] * 40 + [1] * 10)
    return X, y


def test_skf_number_of_folds():
    X, y = make_imbalanced_data()
    assert len(list(StratifiedKFold(n_splits=5).split(X, y))) == 5


def test_skf_each_sample_tested_once():
    X, y = make_imbalanced_data()
    test_idx = np.concatenate([te for _, te in StratifiedKFold(5).split(X, y)])
    np.testing.assert_array_equal(np.sort(test_idx), np.arange(len(y)))


def test_skf_train_test_complementary():
    X, y = make_imbalanced_data()
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y):
        assert set(train_idx).isdisjoint(test_idx)
        assert set(train_idx) | set(test_idx) == set(range(len(y)))


def test_skf_keeps_class_proportions():
    X, y = make_imbalanced_data()
    for _, test_idx in StratifiedKFold(n_splits=5).split(X, y):
        assert np.sum(y[test_idx] == 0) == 8
        assert np.sum(y[test_idx] == 1) == 2


def test_skf_string_labels():
    X, _ = make_imbalanced_data()
    y = np.array(["a"] * 40 + ["b"] * 10)
    for _, test_idx in StratifiedKFold(n_splits=5).split(X, y):
        assert np.sum(y[test_idx] == "b") == 2


def test_skf_accepts_list_y():
    X, y = make_imbalanced_data()
    from_list = list(StratifiedKFold(n_splits=5).split(X, y.tolist()))
    from_array = list(StratifiedKFold(n_splits=5).split(X, y))
    for (tr_a, te_a), (tr_b, te_b) in zip(from_list, from_array):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_skf_same_seed_same_folds():
    X, y = make_imbalanced_data()
    folds_1 = list(StratifiedKFold(shuffle=True, random_state=0).split(X, y))
    folds_2 = list(StratifiedKFold(shuffle=True, random_state=0).split(X, y))
    for (_, te_1), (_, te_2) in zip(folds_1, folds_2):
        np.testing.assert_array_equal(te_1, te_2)


def test_skf_shuffle_changes_folds():
    X, y = make_imbalanced_data()
    folds_1 = list(StratifiedKFold(shuffle=True, random_state=0).split(X, y))
    folds_2 = list(StratifiedKFold().split(X, y))
    assert any(
        not np.array_equal(te_1, te_2) for (_, te_1), (_, te_2) in zip(folds_1, folds_2)
    )


def test_skf_without_y_raises():
    X, _ = make_imbalanced_data()
    with pytest.raises(ValueError, match="precisa de y"):
        list(StratifiedKFold().split(X))


def test_skf_X_y_diff_lengths_raises():
    X, y = make_imbalanced_data()
    with pytest.raises(ValueError, match="mesmo tamanho"):
        list(StratifiedKFold().split(X, y[:-1]))


def test_skf_class_smaller_than_n_splits_raises():
    X, y = make_imbalanced_data()
    with pytest.raises(ValueError, match="classe"):
        list(StratifiedKFold(n_splits=11).split(X, y))


def test_skf_n_splits_less_than_2_raises():
    with pytest.raises(ValueError, match="pelo menos 2"):
        StratifiedKFold(n_splits=1)
