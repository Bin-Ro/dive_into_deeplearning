import torch

A = torch.arange(20.0).reshape(5, 4)
B = torch.ones(4, 3)

print(f'A: {A}')
print(f'B: {B}')
print(f'A.mm(B): {A.mm(B)}')
