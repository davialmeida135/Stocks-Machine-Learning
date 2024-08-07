import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('code/saved_model.keras')

# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df = df[['Close']]

# Initialize the scaler
Ms = MMS()
df[df.columns] = Ms.fit_transform(df)

# Extend the dataset with future dates
future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
future_data = pd.DataFrame(index=future_dates, columns=df.columns)

# Generate synthetic data for future dates (e.g., using the last known values)

future_data['Close'] = df['Close'].iloc[-1]

# Combine the original and future data
extended_df = pd.concat([df, future_data])

# Scale the extended dataset
extended_df[extended_df.columns] = Ms.transform(extended_df)

# Create sequences for prediction
def create_sequence(dataset, lookback=50):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(lookback, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return (np.array(sequences), np.array(labels))

# Split the data into training and testing sets
training_size = round(len(extended_df) * 0.80)
train_data = extended_df[:training_size]
test_data = extended_df[training_size:]

# Create sequences from the training and testing data
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# Predict the initial sequence
test_predicted = model.predict(test_seq)

# Inverse transform the predictions
test_inverse_predicted = Ms.inverse_transform(test_predicted)

# Recursive prediction
num_predictions = 30  # Number of future predictions
lookback = 50
predictions = []

# Use the last sequence from the test data to start the recursive prediction
last_sequence = test_seq[-1]

for _ in range(num_predictions):
    # Predict the next value
    next_pred = model.predict(last_sequence[np.newaxis, :, :])
    next_pred_inverse = Ms.inverse_transform(next_pred)
    predictions.append(next_pred_inverse[0])

    # Update the sequence with the predicted value
    next_sequence = np.append(last_sequence[1:], next_pred, axis=0)
    last_sequence = next_sequence

# Convert predictions to DataFrame
future_dates = pd.date_range(start=extended_df.index[-1], periods=num_predictions + 1, freq='D')[1:]
predicted_df = pd.DataFrame(predictions, columns=['close_predicted'], index=future_dates)

# Save the results to a CSV file
predicted_df.to_csv('code/future_predictions.csv',index=True)