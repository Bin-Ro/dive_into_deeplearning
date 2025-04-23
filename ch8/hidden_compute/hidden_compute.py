import torch
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
print(f'X: {X}')
print(f'W_xh: {W_xh}')
print(f'H: {H}')
print(f'W_hh: {W_hh}')
print(f'X @ W_xh + H @ W_hh: {X @ W_xh + H @ W_hh}')
print(f'torch.cat((X, H), 1) @ torch.cat((W_xh, W_hh), 0): {torch.cat((X, H), 1) @ torch.cat((W_xh, W_hh), 0)}')
