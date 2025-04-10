import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils import data
from torchvision import transforms

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

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../../ch3/data', train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

class Inception(nn.Module):
    #    self 绑定是必须的：只有通过 self.xxx = ... 显式赋值的 nn.Module 或 nn.Parameter 才会被注册到模型的参数列表中。
    #    临时变量会被忽略：在 __init__ 或 forward 中创建的模块或参数若未绑定到 self，则不会被识别为模型的一部分。
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


blk = []

# b1
b = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), 
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk.append(b)

# b2
b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), 
        nn.ReLU(), 
        nn.Conv2d(64, 192, kernel_size=3, padding=1), 
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk.append(b)

# b3
b = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32), 
        Inception(256, 128, (128, 192), (32, 96), 64), 
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk.append(b)

# b4
b = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk.append(b)

# b5
b = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten())
blk.append(b)

net = nn.Sequential(*blk, nn.Linear(1024, 10))
net.to('cuda')

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

num_epochs = 10

trainer = torch.optim.AdamW(net.parameters())
print(f'trainer: {trainer}')
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
