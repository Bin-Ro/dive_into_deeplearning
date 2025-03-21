import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data, index=['row1', 'row2', 'row3'])
print(df, '\n')
print(df.iloc[1, 2], '\n')
print(df.iloc[1], '\n')
print(df.iloc[:2], '\n')
print(df.iloc[:, 1], '\n')
print(df.iloc[[0, 2], [1, 2]], '\n')
print(df.iloc[-1], '\n')
print(df.iloc[:2, [0, 2]], '\n')

# iloc 适用于按 整数索引 选取数据
# 如果想按列名或行标签选取，应该使用 .loc[]
