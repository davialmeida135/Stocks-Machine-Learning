import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy

df = pd.read_csv('2024stocks.csv')

target = df['Will Increase']
features = df[['Day','Month','Day of the Week', 'Open', 'High', 'Low', 'Close', 'Adj Close','close_variation','close_variation%','days_since_last_increase','days_since_last_decrease']]

# Divisão do dataset em treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# Convertendo os datasets para tensores do PyTorch
X_train = torch.from_numpy(X_train.values).float()
X_val = torch.from_numpy(X_val.values).float()
X_test = torch.from_numpy(X_test.values).float()

y_train = torch.from_numpy(y_train.values).float()
y_val = torch.from_numpy(y_val.values).float()
y_test = torch.from_numpy(y_test.values).float()

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
