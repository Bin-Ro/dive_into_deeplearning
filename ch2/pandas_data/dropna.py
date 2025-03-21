import pandas as pd

data = {'A': [1, 2, None, 4, 5], 'B': [None, None, 3, 4, 5], 'C': [1, 2, 3, None, None]}
df = pd.DataFrame(data)
print(df)
missing_counts = df.isna().sum()
column_to_drop = missing_counts.idxmax()
df_cleand = df.drop(columns=[column_to_drop])
print(df_cleand)

threshold = 2
df_cleand = df.dropna(axis=1, thresh=len(df) - threshold + 1)
print(df_cleand)

max_missing = missing_counts.max()
columns_to_drop = missing_counts[missing_counts == max_missing].index
df_cleand = df.drop(columns=columns_to_drop)
print(df_cleand)
