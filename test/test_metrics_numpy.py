import numpy as np
from ml_metrics.metrics_numpy import (F1, MAE, MSE, R2, RMSE, Accuracy,
                                      Precision, Recall)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)


def test_MSE():
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    mse = MSE()
    assert mse(y, y_hat) == mean_squared_error(y, y_hat)


def test_RMSE():
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    rmse = RMSE()
    assert rmse(y, y_hat) == mean_squared_error(y, y_hat) ** 0.5


def test_MAE():
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    rmse = MAE()
    assert rmse(y, y_hat) == mean_absolute_error(y, y_hat)


def test_R2():
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    r2 = R2()
    assert r2(y, y_hat) == r2_score(y, y_hat)


def test_Accuracy():
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    acc = Accuracy()
    assert acc(y, y_hat) == accuracy_score(y, y_hat)


def test_Precision():
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    prec = Precision()
    assert prec(y, y_hat) == precision_score(y, y_hat, average='micro')


def test_Recall():
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    rec = Recall()
    assert rec(y, y_hat) == recall_score(y, y_hat, average='micro')


def test_F1():
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    f1 = F1()
    assert f1(y, y_hat) == f1_score(y, y_hat, average='micro')
    