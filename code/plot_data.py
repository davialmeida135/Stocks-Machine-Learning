import matplotlib.pyplot as plt

from data.data import load_csv
df = load_csv('code/stocks-2.csv')

fig, ax = plt.subplots(2,1,figsize=(10,6))
ax[1].plot(df ['Close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()
plt.show()