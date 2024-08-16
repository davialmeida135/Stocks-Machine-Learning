import pandas as pd
OUT_STEPS = 24
INPUT_WIDTH = 80

df = pd.read_csv('code5/stocks-2.csv', index_col='Date', parse_dates=True)
# Filter the data to include only dates in 2024
#df = df[(df.index.year == 2024) & (df.index.month < 2)]
df = df.loc['2020-06-01':]
#df = df[df.index <= pd.to_datetime('2024-12-31')]
mean = df.mean()
std = df.std()
norm_df = (df - mean) / std

# Split the data
train_df = norm_df.loc[:'2024-02-01']
val_df = norm_df.loc['2024-01-01':'2024-5-31']
test_df = norm_df.loc['2024-02-01':]