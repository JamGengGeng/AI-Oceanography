import numpy as np


class MSE:
    def __init__(self, reduction=None):
        self.reduce = reduction
        
    def __call__(self, y, y_hat):
        MSE = (y - y_hat) ** 2
        match self.reduce:
            case "mean":
                MSE = np.mean(MSE)
            case "sum":
                MSE = np.sum(MSE)
            case _:
                MSE = np.mean(MSE)
        return MSE


class RMSE(MSE):
    def __init__(self, normalize=None):
        self.reduce = None
        self.norm = normalize
        
    def __call__(self, y, y_hat):
        RMSE = super().__call__(y, y_hat) ** 0.5
        match self.norm:
            case "maxmin":
                RMSE /= np.sqrt(RMSE)
            case "mean":
                RMSE /= np.sqrt(RMSE)
            case _:
                pass
        return RMSE


class MAE:
    def __call__(self, y, y_hat):
        MAE = np.mean(np.abs(y - y_hat))
        return MAE


class R2:
    def __call__(self, y, y_hat):
        R2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return R2