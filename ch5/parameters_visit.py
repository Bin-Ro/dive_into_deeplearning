import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
print(f'net: {net}')

print(f'net.state_dict(): {net.state_dict()}', '\n')
print(f'net[0].state_dict(): {net[0].state_dict()}', '\n')

for name, param in net.named_parameters():
    print(name, param)

print('\n')
for name, param in net[0].named_parameters():
    print(name, param)

print(f'net[0].weight: {net[0].weight}, type(net[0].weight): {type(net[0].weight)}')
print(f'net[0].bias: {net[0].bias}, type(net[0].bias): {type(net[0].bias)}', '\n')
print(f'net[0].weight.data: {net[0].weight.data}, type(net[0].weight.data): {type(net[0].weight.data)}')
print(f'net[0].bias.data: {net[0].bias.data}, type(net[0].bias.data): {type(net[0].bias.data)}', '\n')

print(f"net.state_dict()['2.bias']: {net.state_dict()['2.bias']}")
