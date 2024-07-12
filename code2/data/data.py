#####################################################################
# FUNÇÕES PARA CARREGAMENTO E DIVISÃO DOS DADOS PARA TREINO E TESTE #
#####################################################################
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch 
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    target = df['Will Increase']
    features = df[['Day','Month','Day of the Week', 'Open', 'High', 'Low', 'Close', 'Adj Close','close_variation','close_variation%','days_since_last_increase','days_since_last_decrease']]

    return features , target

def prepare_data(features, target, test_size=0.2, val_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.from_numpy(X_train.values).float().to(device)
    X_val = torch.from_numpy(X_val.values).float().to(device)
    X_test = torch.from_numpy(X_test.values).float().to(device)

    y_train = torch.from_numpy(y_train.values).float().to(device)
    y_val = torch.from_numpy(y_val.values).float().to(device)
    y_test = torch.from_numpy(y_test.values).float().to(device)

    return X_train, X_val, X_test, y_train, y_val, y_test, device

def normalize_data(x_train, x_val, x_test, device):
    scaler = MinMaxScaler()  # or StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    return x_train, x_val, x_test