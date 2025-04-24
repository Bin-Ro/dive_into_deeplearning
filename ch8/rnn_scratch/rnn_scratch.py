import torch
from torch import nn
from torch.nn import functional as F

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    W_xh = torch.normal(0, .01, (num_inputs, num_hiddens), device=device, requires_grad=True)
    W_hh = torch.normal(0, .01, (num_hiddens, num_hiddens), device=device, requires_grad=True)
    b_h = torch.zeros(num_hiddens, device=device, requires_grad=True)

    W_hq = torch.normal(0, .01, (num_hiddens, num_outputs), device=device, requires_grad=True)
    b_q = torch.zeros(num_outputs, device=device, requires_grad=True)

    return [W_xh, W_hh, b_h, W_hq, b_q]

def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), H

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

X = torch.arange(10).reshape(2, 5)

num_hiddens = 512
net = RNNModelScratch(28, num_hiddens, 'cuda', get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], 'cuda')
Y, new_state = net(X.to('cuda'), state)
print(f'Y.shape: {Y.shape}')
print(f'new_state.shape: {new_state.shape}')
