import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(f'x: {x}')
print(f'x2: {x2}', '\n')

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x3, y3 = torch.load('x-files')
print(f'x: {x}')
print(f'y: {y}')
print(f'x3: {x3}')
print(f'y3: {y3}', '\n')

my_dict = {'x': x, 'y': y}
torch.save(my_dict, 'my_dict')
my_dict2 = torch.load('my_dict')
print(f'my_dict: {my_dict}')
print(f'my_dict2: {my_dict2}', '\n')
