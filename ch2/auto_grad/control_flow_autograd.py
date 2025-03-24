import torch

def f(x):
    return x if x > 0 else -x

for x in [-1, 1]:
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = f(x)
    y.backward()
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'x.grad: {x.grad}')
    print(f'x.grad == y / x: {x.grad == y / x}', '\n')
