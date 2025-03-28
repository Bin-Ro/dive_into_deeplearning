import torch
from torch import nn

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)

print(f'X.shape: {X.shape}')
print(f'net: {net}')
print(f'net(X).shape: {net(X).shape}')
