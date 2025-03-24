import torch

x = torch.arange(4.0, requires_grad=True)
y = x * x

y.backward(torch.ones_like(x))
print(f'x: {x}')
print(f'y: {y}')
print(f'x.grad: {x.grad}')

x.grad.zero_()
y = x * x
y.sum().backward()
print(f'x: {x}')
print(f'y: {y}')
print(f'x.grad: {x.grad}')

# 非标量输出的 backward()
#
# 当输出是非标量时，PyTorch 不知道如何自动计算梯度，因为梯度是一个向量或矩阵，而不是标量。此时，你需要显式地提供一个 gradient 参数给 backward()，这个参数通常是一个与输出形状相同的张量，表示每个输出元素的权重。
# 原理：
#
#    对于非标量输出，backward() 需要知道如何将输出“缩减”为标量。这是通过提供一个 gradient 参数来实现的。
#
#    gradient 参数的作用是：对输出的每个元素进行加权求和，将非标量输出转换为标量。
#
#    PyTorch 会计算这个加权标量相对于输入的梯度。
