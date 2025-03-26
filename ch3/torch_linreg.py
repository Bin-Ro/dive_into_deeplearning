import torch
import numpy as np
from torch.utils import data
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2

def synthetic_data(w, b, num_examples):
    X = torch.randn(size=(num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, .01, y.shape)
    return X, y.reshape(-1, 1)

features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, .01)
net[0].bias.data.fill_(0)

trainer = torch.optim.SGD(net.parameters(), lr=.03)

num_epochs = 3
batch_size = 10

data_iter = load_array((features, labels), batch_size)

loss = nn.MSELoss()

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.inference_mode():
        print(f'epoch: {epoch}, loss: {loss(net(features), labels)}')
    
print(f'true_w: {true_w}, w: {net[0].weight.data}')
print(f'true_b: {true_b}, b: {net[0].bias.data}')
