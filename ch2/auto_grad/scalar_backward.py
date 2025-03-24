import torch

x = torch.arange(4.0)
print(f'x: {x}')

x.requires_grad_(True)
y = 2 * x.dot(x)
print(f'y: {y}')

y.backward()
print(f'x.grad: {x.grad}')

print(f'x.grad == 4 * x: {x.grad == 4 * x}')

x.grad.zero_()
y = x.sum()
y.backward()
print(f'y: {y}')
print(f'x.grad: {x.grad}')
