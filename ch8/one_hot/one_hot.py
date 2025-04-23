import torch
from torch.nn import functional as F

res = F.one_hot(torch.tensor([0, 2]), 10)
print(res)
res = F.one_hot(torch.randint(0, 10, size=(3, 2)), 10)
print(res.shape)
