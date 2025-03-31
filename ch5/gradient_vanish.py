import torch
from torch import nn
import matplotlib.pyplot as plt

x = torch.arange(-8., 8., .1, requires_grad=True)
y = x.sigmoid()

y.backward(torch.ones_like(y))

plt.plot(x.detach().numpy(), y.detach().numpy(), label='sigmoid')
plt.plot(x.detach().numpy(), x.grad, label='gradient')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('sigmoid')
plt.grid(True)
plt.show()
