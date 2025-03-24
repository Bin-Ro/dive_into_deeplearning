import torch
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 1 / x

x = torch.arange(start=.01, end=2.0, step=.01, dtype=torch.float32)
y = f(x)

x0 = torch.tensor(1.0, requires_grad=True)
y0 = f(x0)
y0.backward()

def df(x):
    return x0.grad * (x - x0) + y0

dfy = df(x)

plt.plot(x, y, label='x**3 - 1 / x')
plt.plot(x, dfy.detach().numpy(), label='tangent')

plt.grid(True)
plt.show()
