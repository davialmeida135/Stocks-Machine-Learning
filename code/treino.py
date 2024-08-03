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

# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename,index_col='Date',parse_dates=True,)
print(df.head())

# Initialize the scaler
Ms = MMS()

# Fit the scaler on the entire dataset
df[df.columns] = Ms.fit_transform(df)


training_size = round(len(df ) * 0.80)

train_data = df [:training_size]
test_data  = df [training_size:]

# Create sequences for prediction
def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(50, len(dataset)): 
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return (np.array(sequences), np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()
# Load the saved model to retrain
model = tf.keras.models.load_model('code/saved_model.keras')

history = model.fit(train_seq, train_label, epochs=150, validation_data=(test_seq, test_label),)

# Save the model
#model.save('code/saved_model.keras')

# Make predictions
test_predicted = model.predict(test_seq)

# Inverse transform the predictions
test_inverse_predicted = Ms.inverse_transform(test_predicted)



num_predictions = test_inverse_predicted.shape[0]
df_slice = df.iloc[-num_predictions:].copy()
# Merging actual and predicted data for better visualization
gs_slic_data = pd.concat([df_slice, pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'], index=df_slice.index)], axis=1)
gs_slic_data[['Open', 'Close']] = Ms.inverse_transform(gs_slic_data[['Open', 'Close']])



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
gs_slic_data.to_csv('code/results.csv')

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# First subplot: Actual vs Predicted Open prices
gs_slic_data[['Open', 'open_predicted']].plot(ax=ax[0])
ax[0].set_xticklabels(ax[0].get_xticks(), rotation=45)
ax[0].set_xlabel('Date', size=15)
ax[0].set_ylabel('Stock Price', size=15)
ax[0].set_title('Actual vs Predicted for Open Price', size=15)
ax[0].legend()

# Second subplot: Actual vs Predicted Close prices
gs_slic_data[['Close', 'close_predicted']].plot(ax=ax[1])
ax[1].set_xticklabels(ax[1].get_xticks(), rotation=45)
ax[1].set_xlabel('Date', size=15)
ax[1].set_ylabel('Stock Price', size=15)
ax[1].set_title('Actual vs Predicted for Close Price', size=15)
ax[1].legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()