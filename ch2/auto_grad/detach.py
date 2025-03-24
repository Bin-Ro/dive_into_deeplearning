import torch

x = torch.arange(4.0, requires_grad=True)

y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(f'x: {x}')
print(f'y: {y}')
print(f'u: {u}')
print(f'z: {z}')
print(f'x.grad: {x.grad}')
print(f'x.grad == u: {x.grad == u}')

x.grad.zero_()
y.sum().backward()
print(f'x.grad: {x.grad}')
