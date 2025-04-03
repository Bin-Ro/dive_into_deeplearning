import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

X = torch.rand(size=(8, 8))
print(f'X: {X.shape}')

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(f'comp_conv2d(conv2d, X).shape: {comp_conv2d(conv2d, X).shape}')

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(f'comp_conv2d(conv2d, X).shape: {comp_conv2d(conv2d, X).shape}')
