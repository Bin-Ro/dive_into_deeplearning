import torch
from torch import nn
from torch.nn import functional as F
import collections
import re
import hashlib
import os
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir='data'):
    assert name in DATA_HUB, f'{name} does not exist in {DATA_HUB}'
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def read_time_machine(cache_dir):
    with open(download('time_machine', cache_dir=cache_dir), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f'Error: Unknown token type: {token}')

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def __iter__(self):
        return iter(self.token_to_idx)


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(cache_dir):
    lines = read_time_machine(cache_dir)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return corpus, vocab


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


corpus, vocab = load_corpus_time_machine(cache_dir='../load_time_machine/data')

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, 'cuda', get_params, init_rnn_state, rnn)

prefix = 'time traveller '
num_preds = 10

state = net.begin_state(batch_size=1, device='cuda')
outputs = [vocab[prefix[0]]]
get_input = lambda: torch.tensor([outputs[-1]], device='cuda').reshape(1, 1)
for y in prefix[1:]:
    _, state = net(get_input(), state)
    outputs.append(vocab[y])
for _ in range(num_preds):
    y, state = net(get_input(), state)
    outputs.append(int(y.argmax(dim=1).reshape(1)))

res = ''.join([vocab.idx_to_token[i] for i in outputs])
print(res)

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
