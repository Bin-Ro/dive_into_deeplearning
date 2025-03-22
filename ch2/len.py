import torch

a = torch.randn(2, 3, 4)
print(f'a: {a}')
print(f'len(a): {len(a)}')
print(f'a.shape: {a.shape}')
print(f'a.sum(dim=0).shape: {a.sum(dim=0).shape}')
print(f'a.sum(dim=1).shape: {a.sum(dim=1).shape}')
print(f'a.sum(dim=2).shape: {a.sum(dim=2).shape}')
