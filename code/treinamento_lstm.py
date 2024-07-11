import torch
from separacao import X_train, X_val, X_test, y_train, y_val, y_test
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import json
#Métricas de treino

input_size = 12
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 1000
batch = 8
is_cuda_available = torch.cuda.is_available()

# Step 2: Set the device to "cuda" if available, else "cpu"
device = torch.device("cuda" if is_cuda_available else "cpu")

X_train = X_train.to(device)
X_val = X_val.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with batch size dimension
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

criterion = nn.BCEWithLogitsLoss()
model = Model(input_size, hidden_size, num_layers, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = [[],[]]
train_losses = []
val_losses = []
# Treinamento do modelo
for epoch in range(num_epochs):
    outputs = model(X_train.unsqueeze(1))  # Adiciona uma dimensão para batch_size
    optimizer.zero_grad()

    loss = criterion(outputs, y_train.unsqueeze(1))  # Adiciona uma dimensão para batch_size
    loss.backward()

    optimizer.step()
    loss_history[0].append(epoch)
    loss_history[1].append(loss.item())
    train_losses.append(loss.item())
    if epoch % 25 == 0 and epoch!=0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        print("Validation Loss: {:.4f}".format(val_loss.item()))

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        val_loss = criterion(y_val_pred, y_val.unsqueeze(1))  # Adiciona uma dimensão para batch_size
        val_losses.append(val_loss.item())
        #print("Validation Loss: {:.4f}".format(val_loss.item()))

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
    'loss_history': loss_history,
    'val_losses':val_losses,
    'train_losses': train_losses,
    'y_val_np': y_val_np.tolist(),
    'y_val_pred_class_np': y_val_pred_class_np.tolist(),
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f'Precision (quantas eu classifiquei como True e eram True): {precision:.4f}')
print(f'Recall (quantas eram True e eu classifiquei como True): {recall:.4f}')
print(f'F1 Score: {f1:.4f}')