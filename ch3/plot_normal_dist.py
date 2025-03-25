import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt

x = torch.arange(start=-5, end=5, step=.01)

dist = Normal(0.0, 1.0)

y = dist.log_prob(x).exp()

plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Normal Distribution')
plt.grid(True)
plt.show()
