import torch
from torch import nn

net = nn.Sequential(nn.Linear(2, 2), nn.ELU(), nn.Linear(2, 2))

def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=1.)
        nn.init.zeros_(m.bias)

def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 42)

net.apply(init_normal)
print(f'net.state_dict(): {net.state_dict()}')

net.apply(init_constant)
print(f'net.state_dict(): {net.state_dict()}')

net[0].apply(init_xavier)
net[2].apply(init_42)
print(f'net.state_dict(): {net.state_dict()}')
