from functools import partial

import torch
import pytest
from ml_metrics.metrics_torch import F1, MAE, MSE, R2, RMSE, Accuracy, Precision, Recall
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def test_RMSE():
    y = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
    y_hat = torch.FloatTensor([6, 5, 4, 2, 7, 8, 9])
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
    y = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
    y_hat = torch.FloatTensor([6, 5, 4, 2, 7, 8, 9])
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
    y = torch.IntTensor([2, 1, 1, 0])
    y_hat = torch.IntTensor([0, 1, 2, 3])
    impl = impl()
    assert impl(y, y_hat) - sk_impl(y, y_hat) < 1e-7
