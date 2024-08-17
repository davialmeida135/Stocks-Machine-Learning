#https://www.tensorflow.org/tutorials/structured_data/time_series?hl=pt-br
import pandas as pd
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
#from model import FeedBack
from window import WindowGenerator
from model import create_multi_conv_model
from var import INPUT_WIDTH, OUT_STEPS
MAX_EPOCHS = 100

multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS)

def compile_and_fit(model, window, patience=2):
    '''early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')'''

    model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        #callbacks=[early_stopping]
                        )
    return history

def train_conv():
    multi_conv_model= create_multi_conv_model(OUT_STEPS, 1)
    compile_and_fit(multi_conv_model, multi_window)
    multi_conv_model.save('code5/conv_model.keras')
    multi_window.plot(multi_conv_model, df = 'train')

train_conv()
#train_conv()
metric_name = 'mean_absolute_error'
'''metric_index = feedback_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()'''
