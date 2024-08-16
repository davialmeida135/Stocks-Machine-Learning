#https://www.tensorflow.org/tutorials/structured_data/time_series?hl=pt-br
import pandas as pd
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from model import FeedBack
from window import WindowGenerator
from model import create_multi_conv_model
from var import INPUT_WIDTH, OUT_STEPS
MAX_EPOCHS = 50

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

def train_feedback():

    multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS)

    feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
    prediction, state = feedback_model.warmup(multi_window.train_batch[0])
    print(prediction.shape)

    history = compile_and_fit(feedback_model, multi_window)
    multi_val_performance={}
    multi_performance = {}
    multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
    multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(feedback_model)
    feedback_model.save('code5/feedback_model.keras')


train_feedback()
