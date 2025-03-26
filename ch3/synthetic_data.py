import torch

def synthetic_data(w, b, num_example):
    X = torch.randn(num_example, len(w))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, label = synthetic_data(true_w, true_b, 1000)
print(f'features.shape: {features.shape}')
print(f'label.shape: {label.shape}')
