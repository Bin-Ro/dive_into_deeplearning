import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1)) if self.use_bias else None

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias if self.use_bias else corr2d(x, self.weight)


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(f'X: {X}')

K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
print(f'Y: {Y}')

conv2d = Conv2D(kernel_size=(1, 2), use_bias=False)
loss = nn.MSELoss()
trainer = torch.optim.AdamW(conv2d.parameters(), lr=1e-1)

for epoch in range(100):
    l = loss(conv2d(X), Y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    print(f'epoch: {epoch + 1}, loss: {l}')

print(f'conv2d.state_dict(): {conv2d.state_dict()}')
