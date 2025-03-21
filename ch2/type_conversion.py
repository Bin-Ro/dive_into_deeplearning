import torch
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
