import torch

x = torch.arange(4, dtype=torch.float32)
print(f'x: {x}')
print(f'x.sum(): {x.sum()}')

A = torch.arange(20.0).reshape(5, 4)
print(f'A: {A}')
print(f'A.shape: {A.shape}')
print(f'A.sum(): {A.sum()}')

A_sum_dim0 = A.sum(dim=0)
print(f'A_sum_dim0: {A_sum_dim0}')
print(f'A_sum_dim0.shape: {A_sum_dim0.shape}')

A_sum_dim1 = A.sum(dim=1)
print(f'A_sum_dim1: {A_sum_dim1}')
print(f'A_sum_dim1.shape: {A_sum_dim1.shape}')

A_sum_dim01 = A.sum(dim=[0, 1])
print(f'A_sum_dim01: {A_sum_dim01}')

print(f'A.mean(): {A.mean()}')
print(f'A.sum() / A.numel(): {A.sum() / A.numel()}')

print(f'A.mean(dim=0): {A.mean(dim=0)}')
print(f'A.sum(dim=0) / A.shape[0]: {A.sum(dim=0) / A.shape[0]}')

sum_A = A.sum(dim=1, keepdim=True)
print(f'sum_A: {sum_A}')
print(f'A / sum_A: {A / sum_A}')

print(f'A.cumsum(dim=0): {A.cumsum(dim=0)}')
