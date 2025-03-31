import torch

M = torch.randn(size=(4, 4))
print(f'M: {M}')

for i in range(99):
    M = M @ torch.randn(size=(4, 4))
print(f'M**100: {M}')
