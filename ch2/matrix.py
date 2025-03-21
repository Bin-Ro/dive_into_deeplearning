import torch

x = torch.arange(4)
print(x)
print(len(x))
print(x.shape)

A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)
