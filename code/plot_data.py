import matplotlib.pyplot as plt

from data.data import load_csv
df = load_csv('code/stocks-2.csv')
fg, ax =plt.subplots(1,2,figsize=(20,7))
ax[0].plot(df ['Open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()
ax[1].plot(df ['Close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()
plt.show()