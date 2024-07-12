############################################################
# GRÁFICOS PLOTADOS A PARTIR DOS DADOS DO TREINO DO MODELO #
############################################################

import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix

with open('train_metrics.json', 'r') as f:
    metrics = json.load(f)

accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']
num_epochs = metrics['num_epochs']
val_losses = metrics['val_losses']
train_losses = metrics['train_losses']
y_val_np = metrics['y_val_np']
y_val_pred_class_np = metrics['y_val_pred_class_np']

def perdas_treino_validacao_3d():
    "Data arrays must have the same length."
    assert len(range(num_epochs)) == len(train_losses) == len(val_losses)
    df = pd.DataFrame()
    df['Epoch'] = range(num_epochs)
    df['train_losses'] = train_losses
    df['val_losses'] = val_losses

    print(df.head(),df.tail())

    fig = px.scatter_3d(df, x='train_losses', y='Epoch', z='val_losses' , color='train_losses',
                        color_continuous_scale='picnic', opacity=0.8,
                        size_max=10, hover_name='Epoch', hover_data=['train_losses', 'Epoch', 'val_losses'])

    fig.update_layout(scene=dict(xaxis_title='train_losses', yaxis_title='Epoch', zaxis_title='val_losses'),
                    title='3D Scatter Plot comparing training loss and validation loss through epochs')

    # Show the plot
    fig.show()

def perdas_treino_validacao():
    # Plotando as perdas de treino e validação
    plt.figure(figsize=(10, 5))
    #sns.lineplot(data=(range(num_epochs), train_losses), marker='*' ,label='Train Loss')
    #sns.lineplot(data=(range(num_epochs) ,val_losses), label='Validation Loss')
    sns.scatterplot(x=range(num_epochs), y=train_losses,label='Train Loss')
    sns.scatterplot(x=range(num_epochs), y=val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.show()

def matriz_confusao():
    cm = confusion_matrix(y_val_np, y_val_pred_class_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def perda_epocas():
    loss = pd.DataFrame()
    loss['Epoch'] = range(num_epochs)
    loss['Loss'] = train_losses
    loss
    sns.lineplot(data=loss, x='Epoch', y='Loss')
    plt.show()

perdas_treino_validacao_3d()