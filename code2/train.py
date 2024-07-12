import torch
import torch.nn as nn
import torch.optim as optim

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

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {loss.item():.5f}")

        # Avaliação do modelo
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
            val_loss = criterion(y_val_pred, y_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
            val_losses.append(val_loss.item())
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Validation Loss: {val_loss.item():.5f}")

    return train_losses, val_losses
