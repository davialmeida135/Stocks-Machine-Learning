import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
OUT_STEPS = 24
INPUT_WIDTH = 64
CONV_WIDTH = 32

scaler = MinMaxScaler()

df = pd.read_csv('code5/stocks-2.csv', index_col='Date', parse_dates=True)
norm_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Filter the data to include only dates in 2024
#df = df[(df.index.year == 2024) & (df.index.month < 2)]
#df = df.loc['2016-01-01':]
#df = df[df.index <= pd.to_datetime('2024-12-31')]
#mean = df.mean()
#std = df.std()
#norm_df = (df - mean) / std

# Split the data
train_df = norm_df.loc[:'2024-01-01']
val_df = norm_df.loc['2024-01-01':]
test_df = norm_df.loc['2023-12-01':]