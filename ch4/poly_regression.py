import math
import numpy as np
import torch
from torch import nn
from torch.utils import data

max_degree = 20

n_train = 100
n_test = 100

features = np.random.normal(size=(n_train + n_test, 1))

true_w = np.zeros(max_degree)
true_w[:4] = np.array([5, 1.2, -3.4, 5.6])

poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)

labels = np.dot(poly_features, true_w)

features, labels, poly_features, true_w = (torch.tensor(x, dtype=torch.float32) for x in (features, labels, poly_features, true_w))

input_shape = 20

train_features = poly_features[:n_train, :input_shape]
train_labels = labels[:n_train]

test_features = poly_features[n_train:, :input_shape]
test_labels = labels[n_train:]

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def evaluate_loss(net, data_iter, loss):
    if isinstance(net, nn.Module):
        net.eval()
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            l = loss(net(X), y)
            metric += torch.tensor([l * y.numel(), y.numel()])
    return metric[0] / metric[1]

train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size=10)
test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size=10)

loss = nn.MSELoss()
net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))

trainer = torch.optim.AdamW(net.parameters())

num_epochs = 4000
for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_acc: {evaluate_loss(net, train_iter, loss)}')
        print(f'epoch: {epoch + 1}, test_acc: {evaluate_loss(net, test_iter, loss)}', '\n')

print(f'true_w: {true_w[:input_shape]}')
print(f'w: {net[0].weight.data}')
print(f'w_error: {(true_w[:input_shape] - net[0].weight.data).square().mean().sqrt()}')
