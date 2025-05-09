import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()

X = torch.rand(2, 20)
print(f'X.shape: {X.shape}')
print(f'net(X).shape: {net(X).shape}')
