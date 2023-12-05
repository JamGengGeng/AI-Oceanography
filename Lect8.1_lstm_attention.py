import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class BoloDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    # 返回数据集大小
    def __len__(self):
        return len(self.data) - self.window_size * 2 + 1

    # 得到数据内容和标签
    def __getitem__(self, idx):
        data_current = self.data[idx : idx + self.window_size]
        data_future = self.data[idx + self.window_size : idx + self.window_size * 2]
        data_current = torch.from_numpy(data_current).float().reshape(-1, 1)
        data_future = torch.from_numpy(data_future).float().reshape(-1, 1)
        return data_current, data_future


class SelfAttention(nn.Module):
    def __init__(self, channels, out_channels, num_heads):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.linear_qkv = nn.Linear(channels, 3 * channels)
        self.linear_out = nn.Linear(channels, out_channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True, dropout=0.1
        )

    def forward(self, x, attn_mask=None):
        x = self.linear_qkv(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        x = self.attn(q, k, v, attn_mask=attn_mask)[0]
        x = self.linear_out(x)
        return x


class AttnLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, layers=3, num_heads=2, use_attn=True
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = SelfAttention(hidden_size, output_size, num_heads=num_heads)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        if self.use_attn:
            b, s, _ = x.shape  # batch_size, sequence_length, feature_number
            x = self.attn(x)
            x = x.view(b, s, -1)
        else:
            x = self.linear(x)
        return x


df = pd.read_excel("博罗.xlsx")
peroid1 = df["s1"].dropna().values / 1e3
peroid2 = df["s2"].dropna().values / 1e3
peroid3 = df["s3"].dropna().values / 1e3
peroid4 = df["s4"].dropna().values / 1e3
peroid5 = df["s5"].dropna().values / 1e3

seed = 42
window_size = 7
batch_size = 256
epochs = 80
lr = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dataloaders = [
    DataLoader(BoloDataset(peroid, window_size), batch_size=batch_size, shuffle=True)
    for peroid in [peroid1, peroid2, peroid3, peroid4]
]
test_dataloader = DataLoader(
    BoloDataset(peroid5, window_size), batch_size=batch_size, shuffle=True
)
model = AttnLSTM(1, 64, 1, layers=3, num_heads=2, use_attn=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

global_step = 0
for epoch in range(epochs):
    # 训练
    model.train()
    # logger.info(f"Epoch: [{epoch+1}/{epochs}]")
    for train_dataloader in train_dataloaders:
        for step, (data_current, data_future) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(data_current.to(device))
            loss = loss_fn(outputs, data_future.to(device))
            loss.backward()
            optimizer.step()

            loss, current = loss.cpu().mean().item(), step * batch_size
            # logger.info(
            #     f"loss: {loss:>5f}  [{current:>5d}/{len(train_dataloader) * batch_size:>5d}]"
            # )
            global_step += 1

    # 评估
    model.eval()
    test_loss = 0
    for data_current, data_future in test_dataloader:
        with torch.no_grad():
            outputs = model(data_current.to(device))
            test_loss += loss_fn(outputs, data_future.to(device))
    test_loss /= len(test_dataloader)
    test_loss = test_loss.cpu().mean().item()
    logger.info(f"[{epoch+1}/{epochs}] Test Avg loss: {test_loss:>5f}")


# 可视化结果
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

idx = 0
x = peroid5[idx : idx + window_size]
y = peroid5[idx + window_size : idx + 2 * window_size]
model.eval()
with torch.no_grad():
    x_in = torch.from_numpy(np.stack([x.reshape(-1, 1)])).float().to(device)
    outputs = model(x_in)
    outputs = outputs[0].cpu().numpy()
ax.plot(np.append(x, y) * 1e3, label="real")
ax.plot(np.append(x, outputs) * 1e3, label="predict")
ax.legend()
plt.show()
