import torch
from torch import nn

shared = nn.Linear(2, 2)

net = nn.Sequential(shared, nn.ELU(), shared)

print(f'net[0] == net[-1]: {net[0] == net[-1]}')
print(f'id(net[0]): {id(net[0])}')
print(f'id(net[-1]): {id(net[-1])}')
