#################################################################
# TRATAMENTO DOS DADOS BAIXADOS E SALVAMENTO DOS DADOS TRATADOS #
#################################################################

import pandas as pd
from download import company_list

close = 'Adj Close'

df = pd.concat(company_list, axis=0)
df.reset_index(inplace=True)

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

if 'Date' in df.columns:
  df['Month'] = df['Date'].dt.month
  df['Day'] = df['Date'].dt.day
  df['Day of the Week'] = df['Date'].dt.dayofweek

column_order = [
    'Date','Close',
]

df = df[column_order].dropna()
print(df.info())
df.set_index('Date', drop=True, inplace=True)

print(df.drop_duplicates().head())

df.to_csv('code5/stocks-2.csv')