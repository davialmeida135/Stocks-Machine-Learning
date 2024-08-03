import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('code/results.csv',index_col='Date',parse_dates=True)

# Plotting the data
df[['Open', 'open_predicted']].plot(figsize=(10, 6))
plt.xticks(rotation=45)
plt.xlabel('Date', size=15)
plt.ylabel('Stock Price', size=15)
plt.title('Actual vs Predicted for Open Price', size=15)
plt.show()