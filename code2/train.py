#######################################################
#  FUNÇÃO QUE TREINA E SALVA MÉTRICAS DO TREINAMENTO  #
#######################################################

from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import json

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=100):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train.unsqueeze(1)) # Adiciona uma dimensão para batch_size
        optimizer.zero_grad()

        loss = criterion(outputs, y_train.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 25 == 0:
            print(f"Epoch: {epoch}, Train Loss: {loss.item():.5f}")

        # Avaliação do modelo
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
            val_loss = criterion(y_val_pred, y_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
            val_losses.append(val_loss.item())
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Validation Loss: {val_loss.item():.5f}")

    
    # Converting predictions to probabilities and binary outputs
    y_val_pred_prob = torch.sigmoid(y_val_pred)
    y_val_pred_class = (y_val_pred_prob > 0.5).float()

    # Accuracy
    accuracy = (y_val_pred_class.squeeze() == y_val).sum().item() / len(y_val)
    print(f'Accuracy: {accuracy:.4f}')

    # Precision, Recall, F1 Score
    y_val_pred_class_np = y_val_pred_class.squeeze().cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    precision = precision_score(y_val_np, y_val_pred_class_np)
    recall = recall_score(y_val_np, y_val_pred_class_np)
    f1 = f1_score(y_val_np, y_val_pred_class_np)
    
    metrics = {
    'num_epochs': num_epochs,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'val_losses':val_losses,
    'train_losses': train_losses,
    'y_val_np': y_val_np.tolist(),
    'y_val_pred_class_np': y_val_pred_class_np.tolist(),
    }

    with open('train_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    return train_losses, val_losses
