import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / (moving_var + eps).sqrt()
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = X.var(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = X.var(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / (var + eps).sqrt()
        moving_mean = momentum * moving_mean + (1. - momentum) * mean
        moving_var = momentum * moving_var + (1. - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        assert num_dims in (2, 4)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
        if self.moving_var.device != X.device:
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=.9)
        return Y

net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), 
        BatchNorm(num_features=6, num_dims=4),
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=2, stride=2), 
        nn.Conv2d(6, 16, kernel_size=5), 
        BatchNorm(num_features=16, num_dims=4),
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=2, stride=2), 
        nn.Flatten(), 
        nn.Linear(16 * 5 * 5, 120), 
        BatchNorm(num_features=120, num_dims=2),
        nn.ReLU(), 
        nn.Linear(120, 84), 
        BatchNorm(num_features=84, num_dims=2),
        nn.ReLU(), 
        nn.Linear(84, 10))

net = net.to('cuda')

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32, device='cuda')
        for X, y in data_iter:
            X, y = X.to('cuda'), y.to('cuda')
            metric += torch.tensor([accuracy(net(X), y), y.numel()], device='cuda')
    return metric[0] / metric[1]

num_epochs = 10

loss = nn.CrossEntropyLoss()

trainer = torch.optim.AdamW(net.parameters())
print(f'trainer: {trainer}')

net.eval()
with torch.inference_mode():
    print(f'train_acc: {evaluate_accuracy(net, train_iter)}')
    print(f'test_acc: {evaluate_accuracy(net, test_iter)}\n')

for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        X, y = X.to('cuda'), y.to('cuda') 
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_accuracy(net, train_iter)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_accuracy(net, test_iter)}\n')
