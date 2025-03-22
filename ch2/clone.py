import torch

X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)
