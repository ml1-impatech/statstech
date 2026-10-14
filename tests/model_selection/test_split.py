import numpy as np
import pytest

from statstech.model_selection.split import train_test_split, KFold, LeaveOneOut

def test_size_float_025():
    X, y = np.arange(10), np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=47)
    assert len(X_test) == 3
    assert len(X_train) == 7

def test_size_int():
    X, y = np.arange(10), np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=4, random_state=47)
    assert len(X_test) == 4
    assert len(X_train) == 6

def test_size_float1_error():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="float entre 0 e 1"):
        train_test_split(X, y, test_size=1.0)

def test_size_float0_error():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="float entre 0 e 1"):
        train_test_split(X, y, test_size=0.0)

def test_size_int10_error():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="inteiro entre 0 e n"):
        train_test_split(X, y, test_size=10)

def test_size_float0_error():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="float entre 0 e 1"):
        train_test_split(X, y, test_size=0.0)

def test_size_int0_error():
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="test_size deve ser um inteiro entre 0 e n"):
        train_test_split(X, y, test_size=0)

def test_size_str_error(): 
    X, y = np.arange(10), np.arange(10)
    with pytest.raises(ValueError, match="float ou um inteiro"):
        train_test_split(X, y, test_size="0.5")

def test_sum_train_test():
    X, y = np.arange(10), np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=47)
    X_concat = np.concatenate([X_train, X_test])
    X_concat.sort()
    assert np.array_equal(X_concat, X)

def test_X_y_aligned():
    X = np.arange(10)
    y = X * 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=47)
    assert np.array_equal(X_train*10, y_train)
    assert np.array_equal(X_test*10, y_test)

def test_random_state_equal():
    X, y = np.arange(100), np.arange(100)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.3, random_state=47)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.3, random_state=47)
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_test1, y_test2)

def test_random_state_different():
    X, y = np.arange(100), np.arange(100)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.3, random_state=47)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.3, random_state=74)
    assert not np.array_equal(X_train1, X_train2)
    assert not np.array_equal(X_test1, X_test2)
    assert not np.array_equal(y_train1, y_train2)
    assert not np.array_equal(y_test1, y_test2)

def test_shuffle_false():
    X, y = np.arange(10), np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)
    assert np.array_equal(X_train, np.arange(7))
    assert np.array_equal(X_test, np.arange(7, 10))

def test_kfold_each_sample_tested_once():
    X = np.arange(13)
    splits = list(KFold(n_splits=5).split(X))

    all_test = np.concatenate([test for _, test in splits])

    np.testing.assert_array_equal(np.sort(all_test), X)

def test_kfold_test_train_complementary():
    X = np.arange(13)

    for train_idx, test_idx in KFold(n_splits=5).split(X):
        assert np.intersect1d(train_idx, test_idx).size == 0
        np.testing.assert_array_equal(np.sort(np.concatenate([train_idx, test_idx])), X)

def test_kfold_num_len_folds():
    X = np.arange(13)
    splits = list(KFold(n_splits=5).split(X))

    assert len(splits) == 5

    len_splits = [len(test) for _, test in splits]

    assert len_splits == [3, 3, 3, 2, 2]