import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.empty(size=(2, 3, 3))
X[0] = torch.arange(9.0).reshape(3, 3)
X[1] = torch.arange(9.0).reshape(3, 3) + 1

K = torch.empty(size=(2, 2, 2))
K[0] = torch.arange(4.0).reshape(2, 2)
K[1] = torch.arange(4.0).reshape(2, 2) + 1

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

K = torch.stack([K, K + 1, K + 2], dim=0)
print(f'K.shape: {K.shape}')

print(f'corr2d_multi_in_out(X, K): {corr2d_multi_in_out(X, K)}')
