import torch
from torch import nn
import copy

class MultiInstanceChain(nn.Module):
    def __init__(self, module, n):
        super().__init__()
        layer = []
        for i in range(n):
            layer.append(copy.deepcopy(module))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        return self.net(X)

module = nn.Linear(2, 2)

net = MultiInstanceChain(module, 3)

print(f'net: {net}')
