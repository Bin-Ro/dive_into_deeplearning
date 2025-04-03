import torch
from torch import nn

X = torch.empty(1, 2, 4, 4)
X[0, 0] = torch.arange(16.0).reshape(4, 4)
X[0, 1] = X[0, 0] + 1
print(f'X: {X}')

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(f'pool2d(X): {pool2d(X)}')
