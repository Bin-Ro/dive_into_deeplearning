import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def cross_entropy(y_hat, y):
    return -y_hat[range(len(y_hat)), y].log()

W = torch.normal(0, .01, size=(784, 10), requires_grad=True)
b = torch.zeros(size=(10,), requires_grad=True)
def net(X):
    return softmax(X.reshape(-1, len(W)) @ W + b)

def evaluate_accuracy(net, data_iter):
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            metric += torch.tensor([accuracy(net(X), y), y.numel()])
    return metric[0] / metric[1]

def sgd(parameters, lr):
    with torch.inference_mode():
        for parameter in parameters:
            parameter -= lr * parameter.grad
            parameter.grad.zero_()

num_epochs = 10

loss = cross_entropy

print(f'train_acc: {evaluate_accuracy(net, train_iter)}')
print(f'test_acc: {evaluate_accuracy(net, test_iter)}', '\n')

for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        sgd([W, b], lr=.1)
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_accuracy(net, train_iter)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_accuracy(net, test_iter)}', '\n')
