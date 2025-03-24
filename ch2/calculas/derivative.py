import torch

def f(x):
    return 3 * x**2 - 4 * x

x = torch.tensor(1.0, requires_grad=True)
print(f'x: {x}')

y = f(x)
print(f'y: {y}')

y.backward()
print(f'x.grad: {x.grad}')

x = torch.tensor(0.0, requires_grad=True)
y = f(x)
y.backward()
print(f'x.grad: {x.grad}')
