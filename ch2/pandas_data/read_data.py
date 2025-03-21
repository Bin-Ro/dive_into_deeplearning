import pandas as pd
import os
import torch

data_file = os.path.join('data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data, '\n')
inputs, outputs = data.iloc[:, :2], data.iloc[:, 2]
print(inputs, '\n')
print(outputs, '\n')

inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].mean())

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs, '\n')
print(outputs, '\n')

X, y = torch.tensor(inputs.values.astype(float), dtype=torch.float32), torch.tensor(outputs.values)
print(X, '\n')
print(y, '\n')
