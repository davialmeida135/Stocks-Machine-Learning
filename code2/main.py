import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import load_data, prepare_data
from model import Model
import torch 
import torch.nn as nn
from train import train_model
from evaluate import evaluate_model, inference

# Carregar os dados
features, target = load_data('/home/rafael/Stocks-Machine-Learning/stocks.csv')

# Preparar os dados
X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data(features, target)

# Definir o modelo e as métricas
input_size = 12
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 100

model = Model(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinar o modelo
train_losses, val_losses = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs)

# Plotar as perdas de treino e validação
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(num_epochs), y=train_losses,label='Train Loss')
sns.lineplot(x=range(num_epochs), y=val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.show()

# Avaliar o modelo
evaluate_model(model, criterion, X_test, y_test)
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
# Salvando os pesos do modelo
torch.save(model.state_dict(), 'modelo_treinado.pth')
print("Modelo salvo com sucesso.")

# Carregando e Usando o Modelo
model_carregado = Model(input_size, hidden_size, num_layers, output_size).to(device)
model_carregado.load_state_dict(torch.load('modelo_treinado.pth'))
model_carregado.eval()
print("Modelo carregado com sucesso.")

# Rodando uma inferência para verificar
#prob_proximo_zero = inference(model_carregado, cenario_proximo_zero)
#print(f"Previsão para o Cenário Próximo de Zero: {prob_proximo_zero:.4f}")
