import torch
from torch import nn
from torch.utils import data

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(size=(num_inputs, 1)) * .01, .05

def synthetic_data(w, b, num_examples):
    X = torch.randn(size=(num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, .01, y.shape)
    return X, y.reshape(-1, 1)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_data = synthetic_data(true_w, true_b, n_train)
test_data = synthetic_data(true_w, true_b, n_test)
train_iter = load_array(train_data, batch_size)
test_iter = load_array(test_data, batch_size, is_train=False)

def linreg(X):
    return X @ w + b

def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)).square().mean()

def sgd(params, lr):
    with torch.inference_mode():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

w = torch.randn(size=(num_inputs, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

net = linreg
loss = square_loss

num_epochs = 100
lr = .003

def l2_penalty(w):
    return w.square().sum()

def evaluate_loss(net, data_iter, loss):
    if isinstance(net, nn.Module):
        net.eval()
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            l = loss(net(X), y)
            metric += torch.tensor([l * y.numel(), y.numel()])
    return metric[0] / metric[1]

print(f'train_loss: {evaluate_loss(net, train_iter, loss)}')
print(f'test_loss: {evaluate_loss(net, test_iter, loss)}', '\n')

lambd = 3

for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y) + lambd * l2_penalty(w)
        l.backward()
        sgd([w, b], lr)
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_loss: {evaluate_loss(net, train_iter, loss)}')
        print(f'epoch: {epoch + 1}, test_loss: {evaluate_loss(net, test_iter, loss)}', '\n')

print(f'w_error: {(true_w - w).square().mean().sqrt()}')
print(f'true_b: {true_b}')
print(f'b: {b}')
