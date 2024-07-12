#######################################################
# AVALIAR MODELO COM CONJUNTO DE TESTES PERSONALIZADO #
#######################################################
from pathlib import Path
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import load_data, prepare_data
from model import Model
import torch 
import torch.nn as nn
from train import train_model
from evaluate import evaluate_model, inference

input_size = 12
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = Path('2024stocks.csv') #Definir origem dos testes, no caso apenas as stocks de 2024
features, target = load_data(datapath)

# Preparar os dados
X_train, X_val, X_test, y_train, y_val, y_test, device = prepare_data(features, target,test_size=1)
criterion = nn.BCEWithLogitsLoss().to(device)


model_path = Path('modelo_treinado.pth')
model_carregado = Model(input_size, hidden_size, num_layers, output_size).to(device)
model_carregado.load_state_dict(torch.load(model_path))
model_carregado.eval()
print("Modelo carregado com sucesso.")
evaluate_model(model_carregado, criterion, X_test, y_test)