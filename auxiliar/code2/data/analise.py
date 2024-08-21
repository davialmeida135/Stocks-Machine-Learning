import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    
    :param filepath: Path to the CSV file
    :return: DataFrame
    """
    df = pd.read_csv(filepath)
    return df

def plot_close_variation_vs_will_increase(df):
    """
    Plot the relationship between close variation and 'Will Increase'.
    
    :param df: DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Will Increase', y='close_variation%', data=df)
    plt.title('Close Variation % vs Will Increase')
    plt.xlabel('Will Increase')
    plt.ylabel('Close Variation %')
    plt.show()

def plot_volume_vs_will_increase(df):
    """
    Plot the relationship between volume and 'Will Increase'.
    
    :param df: DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Will Increase', y='Volume', data=df)
    plt.title('Volume vs Will Increase')
    plt.xlabel('Will Increase')
    plt.ylabel('Volume')
    plt.yscale('log')  # Use logarithmic scale for volume
    plt.show()

def plot_close_prices(df):
    """
    Plot the close prices over time.
    
    :param df: DataFrame
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Close', hue='company_name', data=df)
    plt.title('Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_open_vs_close(df):
    """
    Plot the relationship between open and close prices.
    
    :param df: DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Open', y='Close', hue='Will Increase', data=df, palette='viridis')
    plt.title('Open vs Close Prices')
    plt.xlabel('Open Price')
    plt.ylabel('Close Price')
    plt.show()

def plot_grid(df):
    """
    Create a grid of multiple plots.
    
    :param df: DataFrame
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    sns.boxplot(x='Will Increase', y='close_variation%', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Close Variation % vs Will Increase')
    
    sns.boxplot(x='Will Increase', y='Volume', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Volume vs Will Increase')
    axes[0, 1].set_yscale('log')
    
    sns.lineplot(x='Date', y='Close', hue='company_name', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Close Prices Over Time')
    axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    sns.scatterplot(x='Open', y='Close', hue='Will Increase', data=df, palette='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Open vs Close Prices')
    
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    
    plt.tight_layout()
    plt.show()

def corr_plot(df):
    """
    Plot the correlation matrix of the dataset.
    
    :param df: DataFrame
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr("pearson"), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()


def main():
    # Load the dataset
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'close_variation', 'close_variation%', 'days_since_last_increase', 'days_since_last_decrease','Will Increase']
    filepath = 'stocks.csv'
    df = load_data(filepath)
    
    # Generate individual plots
    #plot_close_variation_vs_will_increase(df)
    #plot_volume_vs_will_increase(df)
    #plot_close_prices(df)
    #plot_open_vs_close(df)
    df = df[df['company_name'] == 'APPLE']
    corr_plot(df[numeric_cols])
    
    # Generate grid of plots
    #plot_grid(df)

if __name__ == "__main__":
    main()
