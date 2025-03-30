import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(2, 10), nn.ELU(), nn.Linear(10, 1))

torch.save(net.state_dict(), 'mlp.params')

net2 = nn.Sequential(nn.Linear(2, 10), nn.ELU(), nn.Linear(10, 1))

net2.load_state_dict(torch.load('mlp.params'))

print(f'net.state_dict(): {net.state_dict()}')
print(f'net2.state_dict(): {net2.state_dict()}')
