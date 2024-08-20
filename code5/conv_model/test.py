import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from window import WindowGenerator
from var import INPUT_WIDTH, OUT_STEPS, test_df

tf.keras.config.enable_unsafe_deserialization()

# Load the saved model
model = tf.keras.models.load_model('code5/conv_model.keras', compile=True)

multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)
'''multi_val_performance={}
multi_performance = {}
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)'''
multi_window.plot(model,max_subplots=3,df=test_df)
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