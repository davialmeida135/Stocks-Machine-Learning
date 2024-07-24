#######################################################
# AVALIAR MODELO COM CONJUNTO DE TESTES PERSONALIZADO #
#######################################################
from pathlib import Path
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

input_size = 12
hidden_size = 32
num_layers = 2
output_size = 1

datapath = Path('stocks2024.csv')
model_path = Path('historico/10kepocas-0.01LR/modelo_treinado.pth')

features, target = load_data(datapath)

# Preparar os dados
X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data(features, target,test_size=0.99)
criterion = nn.BCEWithLogitsLoss().to(device)

#Normalize data if needed
#X_train,X_val,X_test = normalize_data(X_train,X_val,X_test,device)

model_carregado = Model(input_size, hidden_size, num_layers, output_size).to(device)
model_carregado.load_state_dict(torch.load(model_path))
model_carregado.eval()
print("Modelo carregado com sucesso.")
evaluate_model(model_carregado, criterion, X_test, y_test)