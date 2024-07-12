###########################################################
# GRÁFICOS PLOTADOS A PARTIR DOS DADOS DO TESTE DO MODELO #
###########################################################

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix

with open('eval_metrics.json', 'r') as f:
    metrics = json.load(f)

accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']
y_test_np = metrics['y_test_np']
y_test_pred_class_np = metrics['y_test_pred_class_np']

def matriz_confusao():
    cm = confusion_matrix(y_test_np, y_test_pred_class_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

matriz_confusao()