import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(f'X: {X.shape}')
# (8 - 3 + 1 + 2) * (8 - 3 + 1 + 2)
print(f'comp_conv2d(conv2d, X).shape: {comp_conv2d(conv2d, X).shape}')

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(f'comp_conv2d(conv2d, X).shape: {comp_conv2d(conv2d, X).shape}')
