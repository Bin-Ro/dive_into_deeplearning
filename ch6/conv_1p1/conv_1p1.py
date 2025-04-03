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

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

def corr2d_multi_in_out_1p1(X, K):
    assert K.shape[-2:] == (1, 1), f"Expected last two dimensions to be 1, got {K.shape[-2:]}"
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape(c_i, h * w)
    K = K.reshape(c_o, c_i)
    Y = K @ X
    return Y.reshape(c_o, h, w)


X = torch.randn(3, 3, 3)
K = torch.randn(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1p1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(f'Y1: {Y1}')
print(f'Y2: {Y2}')
