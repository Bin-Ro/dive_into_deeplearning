import torch
from torch import nn

# 在 PyTorch 的卷积操作（如 nn.Conv2d）或池化操作（如 nn.MaxPool2d）中，padding 和 stride 参数的顺序是先 height (H) 后 width (W)，即 (H, W)

X = torch.arange(16.0).reshape(1, 1, 4, 4)
print(f'X: {X}')

pool2d = nn.MaxPool2d(3)
print(f'pool2d(X): {pool2d(X)}')

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(f'pool2d(X): {pool2d(X)}')

pool2d = nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3)) 
print(f'pool2d(X): {pool2d(X)}')
