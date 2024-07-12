import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import json

def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        test_loss = criterion(y_test_pred, y_test.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        print(f"Test Loss: {test_loss.item():.4f}")

    # Converting predictions to probabilities and binary outputs
    y_test_pred_prob = torch.sigmoid(y_test_pred)
    y_test_pred_class = (y_test_pred_prob > 0.5).float()

    # Accuracy
    
   
    y_test_pred_class_np = y_test_pred_class.squeeze().cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    

    # Precision, Recall, F1 Score
    accuracy = (y_test_pred_class.squeeze() == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')

    precision = precision_score(y_test_np, y_test_pred_class_np)
    recall = recall_score(y_test_np, y_test_pred_class_np)
    f1 = f1_score(y_test_np, y_test_pred_class_np)

    metrics = {
    'test_loss':test_loss.item(),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'y_test_np': y_test_np.tolist(),
    'y_test_pred_class_np': y_test_pred_class_np.tolist(),
    }

    with open('eval_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
def inference(model, scenario):
    scenario_tensor = torch.from_numpy(scenario.values).float().to(next(model.parameters()).device)
    scenario_tensor = scenario_tensor.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        pred = model(scenario_tensor)
        prob = torch.sigmoid(pred)
        return prob.item()
