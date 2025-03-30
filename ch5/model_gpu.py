import torch
from torch import nn

net = nn.Sequential(nn.Linear(3, 1))
print(f'net: {net}')
print(f'net.state_dict(): {net.state_dict()}', '\n')

net = net.to(device='cuda')
print(f'net: {net}')
print(f'net.state_dict(): {net.state_dict()}', '\n')

x = torch.rand(3, 3, device='cuda')
print(f'x: {x}')
print(f'net(x): {net(x)}')
