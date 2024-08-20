import tensorflow as tf
from tensorflow.keras import models, layers



def create_multi_conv_model(OUT_STEPS,num_features):
    multi_conv_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x[:, -64:, :]),
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(64)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return multi_conv_model

'''def create_multi_conv_model(OUT_STEPS,num_features):
    multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
    return multi_dense_model'''