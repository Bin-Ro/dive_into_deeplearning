import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            metric += torch.tensor([accuracy(net(X), y), y.numel()])
    return metric[0] / metric[1]


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = nn.Parameter(torch.normal(0, .01, size=(num_inputs, num_hiddens), requires_grad=True))
b1 = nn.Parameter(torch.zeros(size=(num_hiddens,), requires_grad=True))
W2 = nn.Parameter(torch.normal(0, .01, size=(num_hiddens, num_outputs), requires_grad=True))
b2 = nn.Parameter(torch.zeros(size=(num_outputs,), requires_grad=True))

def relu(X):
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = relu(X @ W1 + b1)
    return H1 @ W2 + b2

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD([W1, b1, W2, b2], lr=.1)

print(f'train_acc: {evaluate_accuracy(net, train_iter)}')
print(f'test_acc: {evaluate_accuracy(net, test_iter)}', '\n')

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_accuracy(net, train_iter)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_accuracy(net, test_iter)}', '\n')
