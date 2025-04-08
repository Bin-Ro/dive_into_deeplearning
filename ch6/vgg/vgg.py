import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (1, 256), (1, 512), (1, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(.5), nn.Linear(4096, 10))


small_conv_arch = [(pair[0], pair[1] // 4) for pair in conv_arch]

net = vgg(small_conv_arch)
net.to('cuda')

lr = 1e-2
num_epochs = 10
batch_size = 128

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

def evaluate_accuracy(net, data_iter):
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32, device='cuda')
        for X, y in data_iter:
            X, y = X.to('cuda'), y.to('cuda')
            metric += torch.tensor([accuracy(net(X), y), y.numel()], device='cuda')
    return metric[0] / metric[1]

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.AdamW(net.parameters(), lr=lr)

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
