import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('code/saved_model.keras')

# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename,index_col='Date',parse_dates=True)
df = df[['Open', 'Close']]

filename = 'code/stocks2024-2.csv'
df2024 = pd.read_csv(filename,index_col='Date',parse_dates=True)
df2024 = df2024[['Open', 'Close']]

extdf = pd.concat([df, df2024])

# Initialize the scaler
Ms = MMS()

extdf[extdf.columns] = Ms.fit_transform(extdf)


training_size = round(len(df ) * 0.80)

train_data = extdf [:training_size]
test_data  = extdf [training_size:]

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

test_predicted = model.predict(test_seq)

# Inverse transform the predictions
test_inverse_predicted = Ms.inverse_transform(test_predicted)



num_predictions = test_inverse_predicted.shape[0]
df_slice = extdf.iloc[-num_predictions:].copy()
# Merging actual and predicted data for better visualization
df_slic_data = pd.concat([df_slice, pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'], index=df_slice.index)], axis=1)
df_slic_data[['Open', 'Close']] = Ms.inverse_transform(df_slic_data[['Open', 'Close']])

# Save the results to a CSV file
df_slic_data.to_csv('code/results.csv')

