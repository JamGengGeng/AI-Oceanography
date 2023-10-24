import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_metrics.metrics_numpy import MAE, MSE, R2, RMSE


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