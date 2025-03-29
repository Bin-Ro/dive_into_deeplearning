import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

net = nn.Sequential(nn.Linear(3, 5), nn.ELU(), CenteredLayer())

X = torch.rand(3, 3)
print(f'net(X).mean(): {net(X).mean()}')

class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(out_dim, requires_grad=True))

    def forward(self, X):
        return F.relu(X @ self.weight + self.bias)

net = nn.Sequential(MyLinear(2, 2), nn.ELU(), MyLinear(2, 1))

X = torch.rand(3, 2)
print(f'X.shape: {X.shape}')
print(f'net(X).shape: {net(X).shape}')
