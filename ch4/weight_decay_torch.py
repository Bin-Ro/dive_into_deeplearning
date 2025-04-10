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


def evaluate_loss(net, data_iter, loss):
    if isinstance(net, nn.Module):
        net.eval()
    with torch.inference_mode():
        metric = torch.zeros(2, dtype=torch.float32)
        for X, y in data_iter:
            l = loss(net(X), y)
            metric += torch.tensor([l * y.numel(), y.numel()])
    return metric[0] / metric[1]

net = nn.Sequential(nn.Linear(num_inputs, 1))
loss = nn.MSELoss()

trainer = torch.optim.AdamW(net.parameters())
print(f'trainer: {trainer}')

num_epochs = 1000

net.eval()
with torch.inference_mode():
    print(f'train_loss: {evaluate_loss(net, train_iter, loss)}')
    print(f'test_loss: {evaluate_loss(net, test_iter, loss)}', '\n')

for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, train_loss: {evaluate_loss(net, train_iter, loss)}')
        print(f'epoch: {epoch + 1}, test_loss: {evaluate_loss(net, test_iter, loss)}', '\n')

print(f'w_error: {(true_w - net[0].weight.data).square().mean().sqrt()}')
print(f'true_b: {true_b}')
print(f'b: {net[0].bias.data}')
