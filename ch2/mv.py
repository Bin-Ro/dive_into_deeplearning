import torch

A = torch.arange(20.0).reshape(5, 4)
x = torch.arange(4.0)

print(f'A: {A}')
print(f'x: {x}')
print(f'A.mv(x): {A.mv(x)}')
