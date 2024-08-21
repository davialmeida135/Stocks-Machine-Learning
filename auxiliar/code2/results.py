import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


filepath = Path('results.csv')
df = pd.read_csv(filepath)
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
print(df.info())

# Initialize the initial investment
initial_investment = 10000

# Dictionary to store the final investment value for each company
investment_results = {}

# Group by company_name
grouped = df.groupby('company_name')

for company, data in grouped:
    current_investment = initial_investment

    # Sort the data by Date for each company
    data = data.sort_values('Date').reset_index(drop=True)

    # List to store the value of the investment over time for each company
    investment_values = [initial_investment]

    for i in range(1, len(data)):
        if data.loc[i-1, 'Predicted']:
            # Calculate the loss
            percentage_change = (data.loc[i, 'Adj Close'] - data.loc[i-1, 'Adj Close']) / data.loc[i-1, 'Adj Close']
            current_investment *= (1 + percentage_change)  # This will be negative if Adj Close decreases
        investment_values.append(current_investment)

    # Calculate buy-and-hold strategy
    buy_and_hold_values = [initial_investment * (data.loc[i, 'Adj Close'] / data.loc[0, 'Adj Close']) for i in range(len(data))]

    # Store the final investment value for the company
    investment_results[company] = {
        'Final Investment': current_investment,
        'Total Return': current_investment - initial_investment,
        'Investment Values': investment_values,
        'Buy and Hold Values': buy_and_hold_values,
        'Dates': data['Date'].tolist()
    }

# Convert the investment results to a DataFrame for easy analysis
results_df = pd.DataFrame({
    'Company': investment_results.keys(),
    'Final Investment': [v['Final Investment'] for v in investment_results.values()],
    'Total Return': [v['Total Return'] for v in investment_results.values()]
})

print(results_df)

# Plot the investment over time for each company compared to buy-and-hold
plt.figure(figsize=(12, 7))

# Define standard colors
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

for idx, (company, result) in enumerate(investment_results.items()):
    color = colors[idx % len(colors)]
    plt.plot(result['Dates'], result['Investment Values'], label=f"{company} - Predictive Strategy", color=color)
    plt.plot(result['Dates'], result['Buy and Hold Values'], linestyle='--', label=f"{company} - Buy and Hold", color=color)

plt.title('Investment Value Over Time for Each Company')
plt.xlabel('Date')
plt.ylabel('Investment Value')
plt.legend(loc='center left')  # Move the legend to the right
plt.show()