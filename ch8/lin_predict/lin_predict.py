import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32, device='cuda')
x = torch.sin(.01 * time) + torch.normal(0, .2, (T,), device='cuda')

tau = 4
n_train = 600
features = torch.empty(size=(T - tau, tau), device='cuda')
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape(-1, 1)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_iter = load_array((features[:n_train], labels[:n_train]), batch_size=16)

net = nn.Sequential(nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1))
net.to('cuda')

loss = nn.MSELoss()
trainer = torch.optim.AdamW(net.parameters())

net.eval()
with torch.inference_mode():
    print(f'loss: {loss(net(features), labels)}')

for epoch in range(200):
    net.train()
    for X, y in train_iter:
        X, y = X.to('cuda'), y.to('cuda')
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    net.eval()
    with torch.inference_mode():
        print(f'epoch: {epoch + 1}, loss: {loss(net(features), labels)}')

# one step predict
one_step_pred = net(features)

# multistep_pred
multistep_pred = torch.empty(T, device='cuda')
multistep_pred[:n_train + tau] = x[:n_train + tau]

for i in range(n_train + tau, T):
    multistep_pred[i] = net(multistep_pred[i - tau: i].reshape(1, -1))

plt.plot(time[tau:].detach().cpu(), one_step_pred.detach().cpu(), label='one_step_pred')
plt.plot(time.detach().cpu(), x.detach().cpu(), label='sin')
plt.plot(time.detach().cpu(), multistep_pred.detach().cpu(), label='multi step pred')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('sin')
plt.title('lin predict')
plt.legend()
plt.show()
