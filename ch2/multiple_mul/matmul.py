import torch

a = torch.randn(3, 4)
b = torch.randn(4, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a @ b: {a @ b}')
print(f'a.matmul(b): {a.matmul(b)}')

a = torch.arange(24).reshape(2, 3, 4)
b = torch.arange(24).reshape(2, 4, 3)

print(f'a: {a}')
print(f'b: {b}')
print(f'a.matmul(b): {a.matmul(b)}')
print(f'a @ b: {a @ b}')
print(f'a.bmm(b): {a.bmm(b)}')
