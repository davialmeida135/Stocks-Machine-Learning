import matplotlib.pyplot as plt

from data.data import load_csv
df = load_csv('code/stocks-2.csv')

# Create a figure and an axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 'Close' column
ax.plot(df['Close'], label='Close', color='red')

# Set the labels and legend
ax.set_xlabel('Date', size=15)
ax.set_ylabel('Price', size=15)
ax.legend()

# Show the plot
plt.show()