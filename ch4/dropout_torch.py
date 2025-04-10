import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = .2, .5

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

def evaluate_accuracy(net, data_iter):
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            metric += torch.tensor([accuracy(net(X), y), y.numel()])
    return metric[0] / metric[1]

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(dropout2), 
        nn.Linear(num_hiddens2, num_outputs))

num_epochs, batch_size = 10, 256

loss = nn.CrossEntropyLoss()

train_iter, test_iter = load_data_fashion_mnist(batch_size)

trainer = torch.optim.AdamW(net.parameters())

net.eval()
print(f'train_acc: {evaluate_accuracy(net, train_iter)}')
print(f'test_acc: {evaluate_accuracy(net, test_iter)}', '\n')

for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_accuracy(net, train_iter)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_accuracy(net, test_iter)}', '\n')
