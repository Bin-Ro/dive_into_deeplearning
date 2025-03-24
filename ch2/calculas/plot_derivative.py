import torch
import matplotlib.pyplot as plt

def f(x):
    return 3 * x**2 - 4 * x

x = torch.arange(start=-2 + 2 / 3, end=2 + 2 / 3, step=.01, dtype=torch.float32, requires_grad=True)
y = f(x)
y.sum().backward()

plt.plot(x.detach().numpy(), y.detach().numpy(), label='3 * x**2 - 4 * x')
plt.plot(x.detach().numpy(), x.grad, label='6 * x - 4')

plt.legend()

plt.title('deraviative')
plt.xlabel('x')
plt.ylabel('y')

plt.grid(True)

plt.show()
