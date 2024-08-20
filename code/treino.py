import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler as MMS
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
#from data.data import create_sequence
from data.data import load_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import GRU, Dense, Dropout


# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df = df['2016-01-01':'2023-12-31']
print(df.head())

# Initialize the scaler
Ms = MMS()

# Fit the scaler on the entire dataset
mean = df.mean()
std = df.std()
df = (df - mean) / std
df[df.columns] = Ms.fit_transform(df)

training_size = round(len(df) * 0.80)
train_data = df[:training_size]
test_data = df[training_size:]

seq_length = 64

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    start_idx = 0
    for stop_idx in range(seq_length, len(data)):
        sequences.append(data.iloc[start_idx:stop_idx].values)
        labels.append(data.iloc[stop_idx].values)
        start_idx += 1
    return np.array(sequences), np.array(labels)

# Create sequences
train_seq, train_label = create_sequences(train_data, seq_length)
test_seq, test_label = create_sequences(test_data, seq_length)

# Define the model
from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.1))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

# Load the saved model to retrain
# model = tf.keras.models.load_model('code/saved_model.keras')

history = model.fit(train_seq, train_label, epochs=150, validation_data=(test_seq, test_label))

# Save the model
model.save('code/saved_model.keras')

# Combine train and test sequences for prediction
combined_seq = np.concatenate((train_seq, test_seq), axis=0)
combined_label = np.concatenate((train_label, test_label), axis=0)

# Make predictions for the combined dataset
combined_predicted = model.predict(combined_seq)

# Inverse transform the predictions
combined_inverse_predicted = Ms.inverse_transform(combined_predicted)

# Determine the number of predictions
num_predictions = combined_inverse_predicted.shape[0]

# Create a DataFrame for the combined predictions
df_combined = df.iloc[-num_predictions:].copy()
df_combined['close_predicted'] = combined_inverse_predicted

# Inverse transform the actual close prices
df_combined[['Close']] = Ms.inverse_transform(df_combined[['Close']])

# Collecting stats and metrics
stats = {
    "train_loss": history.history['loss'],
    "val_loss": history.history['val_loss'],
    "train_mae": history.history['mean_absolute_error'],
    "val_mae": history.history['val_mean_absolute_error'],
    "test_loss": model.evaluate(test_seq, test_label, verbose=0)[0],
    "test_mae": model.evaluate(test_seq, test_label, verbose=0)[1]
}

# Writing stats and metrics to a JSON file
with open('code/stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

# Save the results to a CSV file
df_combined.to_csv('code/results.csv')

# Create a figure with a single plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot actual vs predicted close prices for the entire dataset
df_combined[['Close', 'close_predicted']].plot(ax=ax, color=['blue', 'red'], label=['Actual', 'Predicted'])

# Draw a vertical line to mark 80% of the data
train_size = len(train_seq)
vertical_line_date = df_combined.index[train_size]
plt.axvline(x=vertical_line_date, color='k', linestyle='--', linewidth=1, label='80% Training Data')

# Customize the plot
ax.set_xticklabels(ax.get_xticks(), rotation=45)
ax.set_xlabel('Date', size=15)
ax.set_ylabel('Stock Price', size=15)
ax.set_title('Actual vs Predicted for Close Price', size=15)
ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()