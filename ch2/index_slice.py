import torch
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print(X[-1])
print(X[1:3])

X[1, 2] = 9
print(X)

X[:2] = 12
print(X)
