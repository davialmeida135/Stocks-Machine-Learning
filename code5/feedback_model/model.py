import tensorflow as tf
from tensorflow.keras import models, layers
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, **kwargs):
        super().__init__(**kwargs)

        self.out_steps = out_steps
        self.units = units
         # Additional Dense layer before LSTM
        self.pre_lstm_dense = tf.keras.layers.Dense(units, activation='relu')
        
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        
        # Dropout layer after LSTM
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        # Additional Dense layer after LSTM
        self.post_lstm_dense = tf.keras.layers.Dense(units, activation='relu')
        
        self.dense = tf.keras.layers.Dense(1)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'out_steps': self.out_steps
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure to pass all arguments from config
        return cls(**{k: v for k, v in config.items() if k not in ['module', 'class_name', 'registered_name']})
