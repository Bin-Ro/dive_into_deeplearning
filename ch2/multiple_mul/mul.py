import torch

a = torch.randn(3, 4)
b = torch.randn(3, 4)

print(f'a: {a}')
print(f'b: {b}')
print(f'a.mul(b): {a.mul(b)}')
print(f'a * b: {a * b}')
