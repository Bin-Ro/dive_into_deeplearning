import torch

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(f'a: {a}')
print(f'X: {X}')
print(f'a + X: {a + X}')
print(f'(a * X).shape: {(a * X).shape}')
