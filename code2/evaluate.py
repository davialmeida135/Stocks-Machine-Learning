import torch
import torch.nn as nn

def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        test_loss = criterion(y_test_pred, y_test.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        print(f"Test Loss: {test_loss.item():.4f}")

def inference(model, scenario):
    scenario_tensor = torch.from_numpy(scenario.values).float().to(next(model.parameters()).device)
    scenario_tensor = scenario_tensor.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        pred = model(scenario_tensor)
        prob = torch.sigmoid(pred)
        return prob.item()
