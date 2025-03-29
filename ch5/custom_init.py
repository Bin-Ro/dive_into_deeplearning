import torch
from torch import nn

net = nn.Sequential(nn.Linear(2, 2), nn.ELU(), nn.Linear(2, 2))

def my_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(f'net.state_dict(): {net.state_dict()}')
