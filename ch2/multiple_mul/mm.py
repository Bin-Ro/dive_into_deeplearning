import torch

a = torch.randn(2, 4)
b = torch.randn(4, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a.mm(b): {a.mm(b)}')
print(f'a @ b: {a @ b}')
