import torch


class MSE:
    def __init__(self, reduction=None):
        self.reduce = reduction

    def __call__(self, y, y_hat):
        MSE = (y - y_hat) ** 2
        match self.reduce:
            case "mean":
                MSE = torch.mean(MSE)
            case "sum":
                MSE = torch.sum(MSE)
            case _:
                MSE = torch.mean(MSE)
        return MSE


class RMSE(MSE):
    def __init__(self, normalize=None):
        self.reduce = None
        self.norm = normalize

    def __call__(self, y, y_hat):
        RMSE = super().__call__(y, y_hat) ** 0.5
        match self.norm:
            case "maxmin":
                RMSE /= torch.sqrt(RMSE)
            case "mean":
                RMSE /= torch.sqrt(RMSE)
            case _:
                pass
        return RMSE


class MAE:
    def __call__(self, y, y_hat):
        MAE = torch.mean(torch.abs(y - y_hat))
        return MAE


class R2:
    def __call__(self, y, y_hat):
        R2 = 1 - torch.sum((y - y_hat) ** 2) / torch.sum((y - torch.mean(y)) ** 2)
        return R2


class ClassifyBase:
    def __call__(self, y, y_hat):
        labels = torch.unique(y_hat)
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        for label in labels:
            self.TP += torch.sum((y_hat == label) & (y == label))
            self.TN += torch.sum((y_hat != label) & (y != label))
            self.FP += torch.sum((y_hat == label) & (y != label))
            self.FN += torch.sum((y_hat != label) & (y == label))


class Accuracy(ClassifyBase):
    def __call__(self, y, y_hat):
        return torch.sum(y_hat == y) / len(y_hat)


class Precision(ClassifyBase):
    def __call__(self, y, y_hat):
        super().__call__(y, y_hat)
        return self.TP / (self.TP + self.FP)


class Recall(ClassifyBase):
    def __call__(self, y, y_hat):
        super().__call__(y, y_hat)
        return self.TP / (self.TP + self.FN)


class F1(ClassifyBase):
    def __call__(self, y, y_hat):
        super().__call__(y, y_hat)
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        return 2 * self.precision * self.recall / (self.precision + self.recall)


class Confusion_Matrix:
    def __call__(self, y, y_hat):
        labels  = torch.unique(y_hat)
        matrix = []
        for i in labels:
            matrix.append([torch.sum((y_hat == i) & (y == j)) for j in labels])
        matrix = torch.Tensor(matrix)
        return matrix.T
