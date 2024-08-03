import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS

# Load the saved model
model = tf.keras.models.load_model('code/saved_model.keras')

# Load the data
filename = 'code/stocks-2.csv'
df = pd.read_csv(filename,index_col='Date',parse_dates=True)
df = df[['Open', 'Close']]

# Initialize the scaler
Ms = MMS()

# Fit the scaler on the data
df_scaled = Ms.fit_transform(df)

# Create sequences for prediction
def create_sequence(data, seq_length=50):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return sequences

# Prepare the input data for prediction
seq_length = 50
input_data = create_sequence(df_scaled, seq_length)
input_data = tf.convert_to_tensor(input_data)

# Make predictions
predictions = model.predict(input_data)

# Inverse transform the predictions
predictions_inverse = Ms.inverse_transform(predictions)

# Print the predictions
print(predictions_inverse)