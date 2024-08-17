import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error
import csv
from var import train_df,val_df,test_df


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                label_columns=None):
        
        # Split the data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                        enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        print('ililililililiili')
        print(inputs,labels)
        return inputs, labels
  
    def plot(self, model=None, plot_col='Close', max_subplots=3, df = 'train'):
        if type(df) == pd.DataFrame:
            inputs,labels = self.custom_batch(df)
            dates = df.index
        elif df == 'test':
            inputs, labels = self.test_batch
            dates = self.test_df.index
        elif df == 'val':
            inputs, labels = self.val_batch
            dates = self.val_df.index
        else:
            inputs, labels = self.train_batch
            dates = self.train_df.index

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        print('=============================')
        #print(inputs)
        # Initialize lists to store metrics
        metrics = []
        r2_scores = []
        mae_scores = []

        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            
            # Plot input sequences
            plt.plot(dates[self.input_indices], inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # Plot labels
            plt.scatter(dates[self.label_indices], labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                # Plot predictions
                print(inputs)
                predictions = model(inputs)
                print(f"dates[self.label_indices].shape: {dates[self.label_indices].shape}")
                print(f"predictions[n, :, label_col_index].shape: {predictions[n, :, label_col_index].shape}")

                plt.scatter(dates[self.label_indices], predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
                
                # Extract metrics
                actuals = labels[n, :, label_col_index]
                preds = predictions[n, :, label_col_index]
                for i, date in enumerate(dates[self.label_indices]):
                    actual = actuals[i].numpy()  # Convert to NumPy scalar
                    predicted = preds[i].numpy()  # Convert to NumPy scalar
                    metrics.append([date, actual, predicted])

                # Convert TensorFlow tensors to NumPy arrays
                actuals_np = actuals.numpy()
                preds_np = preds.numpy()
                # Calculate R² and MAE
                r2 = r2_score(actuals_np, preds_np)
                mae = mean_absolute_error(actuals_np, preds_np)
                r2_scores.append(r2)
                mae_scores.append(mae)

            if n == 0:
                plt.legend()

        plt.xlabel('Date')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        plt.show()
        # Save metrics to a CSV file
        with open('code5/metrics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Actual', 'Predicted'])
            writer.writerows(metrics)

        # Save R² and MAE scores to a CSV file
        with open('code5/metrics_summary.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['R2', 'MAE'])
            for r2, mae in zip(r2_scores, mae_scores):
                writer.writerow([r2, mae])
    
    def make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,)

        print('dsdsdsdsdsdsdsdsdsdsdsds')
        print(ds)
        ds = ds.map(self.split_window)
        print('MAPMAMAPMAMAPMA')
        print(ds)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df,shuffle=True)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)
    
    @property
    def train_batch(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        print('===++++++++++++++++==========')
        print(result)
        return result
    
    @property
    def val_batch(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.val))
            # And cache it for next time
            self._example = result
        print('===++++++++++++++++==========')
        print(result)
        return result

    @property
    def test_batch(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        print('===++++++++++++++++==========')
        print(result)
        return result
    
    def custom_batch(self, df):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.make_dataset(df,shuffle=False)))
            # And cache it for next time
            self._example = result
        print('===++++++++++++++++==========')
        print(result)
        return result


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])