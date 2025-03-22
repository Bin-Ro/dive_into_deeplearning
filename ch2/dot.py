import torch

x = torch.arange(4.0)
y = torch.ones(4, dtype=torch.float32)

print(f'x: {x}')
print(f'y: {y}')
print(f'x.dot(y): {x.dot(y)}')
print(f'(x * y).sum(): {(x * y).sum()}')
print(f'x @ y: {x @ y}')
