import torch

x = torch.arange(4, device='cuda')
y = torch.arange(4, 8, device='cpu')
print(f'x: {x}, x.device: {x.device}')
print(f'y: {y}, y.device: {y.device}')

z = y.cuda()
print(f'x + z: {x + z}')
