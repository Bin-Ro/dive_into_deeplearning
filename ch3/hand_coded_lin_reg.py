import torch
import random

def synthetic_data(w, b, num_examples):
    X = torch.randn(size=(num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, .01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2

num_examples = 1000

features, labels = synthetic_data(true_w, true_b, num_examples)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        ind = torch.tensor(index[i: min(i + batch_size, num_examples)])
        yield features[ind], labels[ind]

w = torch.normal(0, .01, size=(2,), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X):
    return X @ w + b

def square_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape))**2).mean()

def sgd(params, lr):
    with torch.inference_mode():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

loss = square_loss
net = linreg

num_epochs = 3
lr = .03
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X), y)
        l.backward()
        sgd([w, b], lr)
    with torch.inference_mode():
        print(f'epoch: {epoch}, loss: {loss(net(features), labels)}')

print(f'true_w: {true_w}, w: {w}')
print(f'true_b: {true_b}, b: {b}')
