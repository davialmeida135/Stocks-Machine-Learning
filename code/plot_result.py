import matplotlib.pyplot as plt
import pandas as pd
def plot_normal():
    df=pd.read_csv('code/results.csv',index_col='Date',parse_dates=True)
    vertical_line_position = int(len(df) * 0.80)
    vertical_line_date = df.index[vertical_line_position]
    df = df[df.index >= pd.to_datetime('2000-01-01')]

    # Plotting the data
    ax = df[['Close', 'close_predicted']].plot(figsize=(10, 6))
    plt.axvline(x=vertical_line_date, color='k', linestyle='--', linewidth=1, label='80% Training Data')

    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.title('Actual vs Predicted for Close Price', size=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()  # Enable minor ticks
    plt.show()

def plot_forecast():
    # Read the predicted data
    predicted_df = pd.read_csv('code/future_predictions.csv', index_col='Date', parse_dates=True)
    
    # Read the actual data
    actual_df = pd.read_csv('code/stocks2024-2.csv', index_col='Date', parse_dates=True)
    
    # Merge the datasets on the date index
    merged_df = predicted_df.merge(actual_df, left_index=True, right_index=True, suffixes=('_predicted', '_actual'))
    
    # Plotting the data
    ax = merged_df[['close_predicted', 'Close']].plot(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.title('Predicted vs Actual for Open Price', size=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()  # Enable minor ticks
    plt.show()

plot_normal()
#plot_forecast()