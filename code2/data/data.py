#####################################################################
# FUNÇÕES PARA CARREGAMENTO E DIVISÃO DOS DADOS PARA TREINO E TESTE #
#####################################################################
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch 
from sklearn.model_selection import train_test_split
import numpy as np
import json

def load_data(file_path):
    df = pd.read_csv(file_path)
    target = df['Will Increase']
    features = df[['Day','Month','Day of the Week', 'Open', 'High', 'Low', 'Close', 'Adj Close','close_variation','close_variation%','days_since_last_increase','days_since_last_decrease']]

    return features , target

def prepare_data(features, target, test_size=0.2, val_size=0.25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if test_size == 1:
        # Assuming 'target' is your pandas DataFrame
        X_test=torch.from_numpy(features.to_numpy()).float().to(device)
        y_test=torch.from_numpy(target.to_numpy()).float().to(device)
        return None, None, X_test, None, None, y_test, device
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    X_train = torch.from_numpy(X_train.values).float().to(device)
    X_val = torch.from_numpy(X_val.values).float().to(device)
    X_test = torch.from_numpy(X_test.values).float().to(device)

    y_train = torch.from_numpy(y_train.values).float().to(device)
    y_val = torch.from_numpy(y_val.values).float().to(device)
    y_test = torch.from_numpy(y_test.values).float().to(device)

    return X_train, X_val, X_test, y_train, y_val, y_test, device

def normalize_data(x_train, x_val=None, x_test=None, device=None):
    scaler = MinMaxScaler()  # or StandardScaler()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_train = scaler.fit_transform(x_train) 
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)

    if x_val is not None:
        x_val = scaler.transform(x_val)
        x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    if x_test is not None:
        x_test = scaler.transform(x_test)    
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    return x_train, x_val, x_test

def merge_data(file_path):
    df = pd.read_csv(file_path)
    
    with open('eval_metrics.json', 'r') as f:
        metrics = json.load(f)

    y_test_pred_class_np = metrics['y_test_pred_class_np']
    if len(y_test_pred_class_np) != df.shape[0]:
        print("Predição de tamanho diferente do dataset")
        return None
    
    if 'Will Increase' in df.columns:
        will_increase = df.pop('Will Increase')  # Remove the column and store it
        df['Will Increase'] = will_increase

    df['Predicted'] = y_test_pred_class_np
    df['Predicted'] = df['Predicted'].map({0: False, 1: True})
    df = df.iloc[:, 0:]
    print(df.tail())
    df.to_csv('results.csv',index=False)
    return df