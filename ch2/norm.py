import torch

u = torch.tensor([3.0, -4])
print(f'u: {u}')
print(f'u.norm(): {u.norm()}')
print(f'u.abs().sum(): {u.abs().sum()}')
print(f'torch.ones(4, 9).norm(): {torch.ones(4, 9).norm()}')
