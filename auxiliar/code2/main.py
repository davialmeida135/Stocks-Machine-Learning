################################
# TREINO E AVALIAÇÃO DO MODELO #
################################

import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data.data import load_data, normalize_data, prepare_data
from model import Model
import torch 
import torch.nn as nn
from train import train_model
from evaluate import evaluate_model, inference

# Carregar os dados
datapath = Path('stocks.csv')
features, target = load_data(datapath,'APPLE')

# Preparar os dados
X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data(features, target)

input_size = features.shape[1]
# Definir o modelo e as métricas
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 2000

model = Model(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Normalize the features
#X_train,X_val,X_test = normalize_data(X_train,X_val,X_test,device)

# Treinar o modelo
train_losses, val_losses = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs)

# Avaliar/Testar o modelo
evaluate_model(model, criterion, X_test, y_test)

# Salvando os pesos do modelo
model_path = Path('modelo_treinado.pth')
torch.save(model.state_dict(), model_path)
print("Modelo salvo com sucesso.")

'''
# Inferência para cenários
cenario_proximo_zero = pd.DataFrame({
    'Year': [2024],
    'Open': [features['Open'].mean()],
    'High': [features['High'].mean()],
    'Low': [features['Low'].mean()],
    'Close': [features['Close'].mean()],
    'Adj Close': [features['Adj Close'].mean()]
})

prob_proximo_zero = inference(model, cenario_proximo_zero)
print(f"Previsão para o Cenário Próximo de Zero: {prob_proximo_zero:.4f}")
'''

'''
# Carregando e Usando o Modelo
model_path = Path('modelo_treinado.pth')
model = Model(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Modelo carregado com sucesso.")
'''
# Rodando uma inferência para verificar
#prob_proximo_zero = inference(model_carregado, cenario_proximo_zero)
#print(f"Previsão para o Cenário Próximo de Zero: {prob_proximo_zero:.4f}")
