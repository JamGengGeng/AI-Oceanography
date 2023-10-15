from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from scipy.io import loadmat
from loguru import logger


class LeakyRelu:
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha
    
    def __repr__(self) -> str:
        return f"LeakyRelu({self.alpha})"
    
    def forward(self, x):
        self.x = x
        return np.maximum(self.alpha * x, x)
    
    def backward(self, dout):
        dout[self.x < 0] *= self.alpha
        return dout


class Relu:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Relu()"

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.x < 0] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def __repr__(self) -> str:
        return "Sigmoid()"

    def forward(self, x):
        out = self.sigmoid(x)
        self.out = x
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class Linear:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.rand(in_channels, out_channels)
        self.b = np.random.rand(out_channels)

        self.x = None
        self.dW = None
        self.db = None

    def __repr__(self) -> str:
        return f"W: {self.W.shape}, b: {self.b.shape}"

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class LiNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: list[int],
        output_size: int,
        activation: Relu | Sigmoid = Relu,
    ):
        self.layers = OrderedDict()
        self.layers["linear_1"] = Linear(input_size, hidden_size[0])
        self.layers["activation_1"] = activation()
        for i in range(1, len(hidden_size)):
            self.layers["linear_" + str(i + 1)] = Linear(
                hidden_size[i - 1], hidden_size[i]
            )
            self.layers["activation_" + str(i + 1)] = activation()
        self.layers["linear_output"] = Linear(hidden_size[-1], output_size)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self) -> str:
        return str(model.layers)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def backward(self, loss):
        # backward
        dout = loss

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


class MSELoss:
    def __call__(self, input, target):
        self.input = input
        self.target = target
        return np.sum((input - target) ** 2) / np.prod(self.input.shape)

    def backward(self):
        return 2 * (self.input - self.target) / np.prod(self.input.shape)


class Optimizer:
    def __init__(self, model, lr) -> None:
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers.values():
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


def get_data():
    # read dataset 1
    Dataset = nc.Dataset(r"Data/OPEN-OHCv1.1.1_1993-2022.nc")
    ohc = Dataset["OHC"][:312, 3, 25:155, :]
    Dataset.close()
    gohc = np.nansum(ohc, axis=(1, 2))
    # read dataset 2
    Dataset = loadmat(r"Data/SST1993-2018.mat")
    sst = Dataset["var"][:]
    sst = np.flip(sst, 1)
    gsst = np.nanmean(sst, axis=(1, 2))

    x = gsst - 17.2
    x = np.array([x, x**2, x**3]).T
    # x = np.array([x]).T
    y = (gohc / 1e4 - 1.365) * 30

    return x, y


# 生成数据集
x_all, y_all = get_data()
y_all = y_all.reshape(y_all.shape[0], -1)
model = LiNet(input_size=3, hidden_size=[4, 8, 4], output_size=1, activation=LeakyRelu)
print(model)
loss_fn = MSELoss()
optimizer = Optimizer(model, lr=1e-3)

epochs = 25000
loss_list = []

# train loop
for i in range(epochs):
    y = model(x_all)
    loss = loss_fn(y, y_all)
    loss_grad = loss_fn.backward()
    model.backward(loss_grad)
    optimizer.step()
    loss_list.append(loss)
    if (i + 1) % 1000 == 0:
        logger.info(f"Epoch: [{i+1}/{epochs}], loss: {loss}")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(loss_list)
ax.set_ylim(0, 0.1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y, y_all)
plt.show()
