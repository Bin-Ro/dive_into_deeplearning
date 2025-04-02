import torch

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.arange(9.0).reshape(3, 3)
K = torch.arange(4.0).reshape(2, 2)

print(f'corr2d(X, K): {corr2d(X, K)}')
