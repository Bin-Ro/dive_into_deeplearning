import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), 
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), 
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(), 
        nn.Linear(6400, 4096), nn.ReLU(), 
        nn.Dropout(p=.5), 
        nn.Linear(4096, 4096), nn.ReLU(), 
        nn.Dropout(p=.5), 
        nn.Linear(4096, 10))

net = net.to('cuda')

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

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

batch_size = 128

train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr = 1e-3
num_epochs = 10

trainer = torch.optim.AdamW(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

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
