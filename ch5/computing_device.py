import torch
from torch import nn

print(f"torch.device('cpu'): {torch.device('cpu')}")
print(f"torch.device('cuda'): {torch.device('cuda')}")
print(f"torch.device('cuda:1'): {torch.device('cuda:1')}")

print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

def try_gpu(i=0):
    return torch.device(f'cuda:{i}') if torch.cuda.device_count() >= i + 1 else torch.device('cpu') 

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(f'try_gpu(): {try_gpu()}')
print(f'try_gpu(10): {try_gpu(10)}')
print(f'try_all_gpus(): {try_all_gpus()}')
