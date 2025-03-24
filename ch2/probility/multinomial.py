import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt

fair_probs = torch.ones(6) / 6

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=-1, keepdim=True)

for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=f'P(die={i})')

plt.grid(True)
plt.legend()
plt.xlabel('Groups of experiments')
plt.xlabel('Estimated probability')
plt.show()
