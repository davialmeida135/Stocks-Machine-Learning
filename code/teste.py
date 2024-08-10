from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('code/saved_model.keras')

# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename,index_col='Date',parse_dates=True)
df = df[['Close']]

filename = 'code/stocks2024-2.csv'
df2024 = pd.read_csv(filename,index_col='Date',parse_dates=True)
df2024 = df2024[['Close']]

extdf = pd.concat([df, df2024])

# Initialize the scaler
Ms = MMS()

extdf[extdf.columns] = Ms.fit_transform(extdf)


training_size = round(len(df ) * 0.80)

train_data = extdf [:training_size]
test_data  = extdf [training_size:]

seq_lenght = 16

# Create sequences for prediction
def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(seq_lenght, len(dataset)): 
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return (np.array(sequences), np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)
ext_seq, ext_label = create_sequence(extdf)

# Combine train and test sequences for prediction
combined_seq = np.concatenate((train_seq, test_seq), axis=0)
combined_label = np.concatenate((train_label, test_label), axis=0)

# Make predictions for the combined dataset
combined_predicted = model.predict(combined_seq)

# Inverse transform the predictions
combined_inverse_predicted = Ms.inverse_transform(combined_predicted)

# Determine the number of predictions
num_predictions = combined_inverse_predicted.shape[0]

print(num_predictions)

# Adjust the DataFrame index to match the number of predictions
df_combined = extdf.iloc[-num_predictions:].copy()
print(len(df_combined))
df_combined['close_predicted'] = combined_inverse_predicted[:, 0]  # Assuming 'Close' is the first column

# Inverse transform the actual close prices
df_combined[['Close']] = Ms.inverse_transform(df_combined[['Close']])

# Collecting stats and metrics

# Save the results to a CSV file
df_combined.to_csv('code/results.csv')
import matplotlib.dates as mdates

df_combined = pd.read_csv('code/results.csv',index_col='Date',parse_dates=True)
# Create a figure with a single plot
fig, ax = plt.subplots(figsize=(10,5))

# Plot actual vs predicted close prices for the entire dataset
df_combined[['Close', 'close_predicted']].plot(ax=ax, color=['blue', 'red'], label=['Actual', 'Predicted'])
# Set the date format on the X-axis
# Set the date format on the X-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

# Rotate and align the tick labels so they look better
fig.autofmt_xdate()

ax.set_xlabel('Date', size=15)
ax.set_ylabel('Stock Price', size=15)
ax.set_title('Actual vs Predicted for Close Price', size=15)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()