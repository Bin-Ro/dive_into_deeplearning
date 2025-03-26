import torch

def synthetic_data(w, b, num_examples):
    X = torch.randn(num_examples, len(w))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        ind = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[ind], labels[ind]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(f'X: {X}')
    print(f'y: {y}', '\n')
