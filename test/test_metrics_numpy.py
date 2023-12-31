from functools import partial

import numpy as np
import pytest
from ml_metrics.metrics_numpy import F1, MAE, MSE, R2, RMSE, Accuracy, Precision, Recall, Confusion_Matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    confusion_matrix
)


def test_RMSE():
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    rmse = RMSE()
    assert rmse(y, y_hat) == mean_squared_error(y, y_hat) ** 0.5


@pytest.mark.parametrize(
    "impl, sk_impl",
    [
        (MAE, mean_absolute_error),
        (MSE, mean_squared_error),
        (R2, r2_score),
    ],
)
def test_regression(impl, sk_impl):
    y = np.array([1, 2, 3, 4, 5, 6, 7])
    y_hat = np.array([6, 5, 4, 2, 7, 8, 9])
    impl = impl()
    assert impl(y, y_hat) - sk_impl(y, y_hat) < 1e-7


@pytest.mark.parametrize(
    "impl, sk_impl",
    [
        (Accuracy, accuracy_score),
        (Precision, partial(precision_score, average="micro")),
        (Recall, partial(recall_score, average="micro")),
        (F1, partial(f1_score, average="micro")),
    ],
)
def test_classification(impl, sk_impl):
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    impl = impl()
    assert impl(y, y_hat) - sk_impl(y, y_hat) < 1e-7


def test_confusion_matrix():
    y = np.array([2, 1, 1, 0])
    y_hat = np.array([0, 1, 2, 3])
    cm = Confusion_Matrix()
    assert np.array_equal(cm(y, y_hat), confusion_matrix(y, y_hat))
