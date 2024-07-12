import pandas as pd
from download import company_list

close = 'Adj Close'

df = pd.concat(company_list, axis=0)
df.reset_index(inplace=True)
df.sort_values(by=['company_name','Date'])
df['Will Increase'] = df.groupby('company_name')[close].shift(-1) > df.groupby('company_name')[close].shift(0)
df['close_variation'] = df.groupby('company_name')[close].shift(-1) - df.groupby('company_name')[close].shift(0)
df['close_variation%'] = df.groupby('company_name')[close].shift(-1) / df.groupby('company_name')[close].shift(0)

def calculate_days_since(df):
    last_increase = 0
    last_decrease = 0
    days_since_increase = []
    days_since_decrease = []
    for will_increase in df['Will Increase']:
        days_since_increase.append(last_increase)
        days_since_decrease.append(last_decrease)
        if will_increase:
            last_increase = 0
            last_decrease += 1
        else:
            last_increase += 1
            last_decrease = 0
    return days_since_increase, days_since_decrease


for company, group in df.groupby('company_name'):
    days_since_increase, days_since_decrease = calculate_days_since(group)
    df.loc[group.index, 'days_since_last_increase'] = days_since_increase
    df.loc[group.index, 'days_since_last_decrease'] = days_since_decrease

if 'Date' in df.columns:
  df['Month'] = df['Date'].dt.month
  df['Day'] = df['Date'].dt.day
  df['Day of the Week'] = df['Date'].dt.dayofweek

column_order = [
    'Month','Day', 'Day of the Week', 'Open', 'High', 'Low', 'Close',
    'Adj Close', 'Volume', 'company_name', 'Will Increase', 'close_variation',
    'close_variation%', 'days_since_last_increase', 'days_since_last_decrease'
]

df = df[column_order].dropna()
print(df.info())

print(df.sort_values(by=['company_name',"Month","Day"]).head(15))

df.to_csv('2024stocks.csv', index=False)