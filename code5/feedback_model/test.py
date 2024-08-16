import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import FeedBack
from window import WindowGenerator
from var import INPUT_WIDTH, OUT_STEPS

tf.keras.config.enable_unsafe_deserialization()

df = pd.read_csv('code5/stocks-2.csv', index_col='Date', parse_dates=True)
    

# Filter the data to include only dates in 2024
#df = df[(df.index.year == 2024) & (df.index.month < 2)]
df = df.loc['2024-02-01':]
#df = df[df.index <= pd.to_datetime('2024-12-31')]
mean = df.mean()
std = df.std()
df = (df - mean) / std

# Load the saved model
model = tf.keras.models.load_model('code5/feedback_model.keras', custom_objects={'FeedBack': FeedBack})

multi_window = WindowGenerator(input_width=64,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)
'''multi_val_performance={}
multi_performance = {}
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)'''
multi_window.plot(model,max_subplots=1,df=df)
#feedback_model.save('code5/feedback_model.keras')


'''x = np.arange(len(multi_performance))
width = 0.3

print(multi_performance)
metric_name = 'mean_absolute_error'
metric_index = feedback_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()'''