import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.empty(size=(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.arange(9.0).reshape(3, 3)
print(f'X: {X}')
print(f'pool2d(X, (2, 2)): {pool2d(X, (2, 2))}')
print(f"pool2d(X, (2, 2), 'avg'): {pool2d(X, (2, 2), 'avg')}")
