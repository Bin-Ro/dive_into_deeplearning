import torch
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a)
print(b)
print(a + b)

A = torch.randn(3, 4)
B = torch.randn(5, 4)
dist = torch.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(dim=2))
print(dist.shape)

X = torch.rand(100, 5)
mean = X.mean(dim=0)
std = X.std(dim=0)

X_norm = (X - mean) / std
print(X_norm.shape)

# 规则	说明
# 从后往前对齐	两个 Tensor 形状从右侧开始对齐，左侧不足的补 1
# 维度匹配	维度相等，或其中一个是 1，才可以广播
# 不能匹配时报错	不能广播的 Tensor 不能进行计算
# 标量可以广播	标量 Tensor 可以广播到任何形状
