import matplotlib.pyplot as plt
import seaborn as sns
import json
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']
num_epochs = metrics['num_epochs']
loss_history = metrics['loss_history']
val_losses = metrics['val_losses']
train_losses = metrics['train_losses']
y_val_np = metrics['y_val_np']
y_val_pred_class_np = metrics['y_val_pred_class_np']

def perdas_treino_validacao():
    "Data arrays must have the same length."
    print(len(loss_history[0]))
    print(len(val_losses))
    assert len(range(num_epochs)) == len(loss_history[0]) == len(val_losses)



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

perdas_treino_validacao()