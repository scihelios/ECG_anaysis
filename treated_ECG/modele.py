import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPredictorV2(nn.Module):
    def __init__(self):
        super(GaussianPredictorV2, self).__init__()
        self.fc1 = nn.Linear(400, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 15)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
def load_data(folder_path):
    all_inputs = []
    all_targets = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            print(file_name)
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data = json.load(file)
                signals = data["signal"]
                params = data["gaussienne"]
                all_inputs.append(torch.tensor(signals))
                all_targets.append(torch.tensor(params))

    
    all_inputs = torch.stack(all_inputs)
    all_targets = torch.stack(all_targets)
    return all_inputs, all_targets

# Exemple : remplacer 'your_folder_path' par le chemin de votre dossier
inputs, targets = load_data('C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ECG')

from sklearn.model_selection import train_test_split

# Assuming 'all_inputs' and 'all_targets' are your complete dataset
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42)  # 20% data for validation

from torch.utils.data import DataLoader, TensorDataset

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

from torch.optim.lr_scheduler import StepLR

# Assuming GaussianPredictor, dataloader, and validation_dataloader are already defined

model = GaussianPredictorV2()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Adjust the step size and gamma as needed

num_epochs = 3000
best_val_loss = float('inf')
patience, trials = 100, 0  # Early stopping parameters


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.float(), targets.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_training_loss = total_loss / len(train_dataloader)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs, targets = inputs.float(), targets.float()
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(validation_dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        trials = 0
    else:
        trials += 1

    # Early stopping
    if trials >= patience:
        print("Early stopping triggered")
        break

    scheduler.step()  # Adjust the learning rate

'''
model = GaussianPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('on')

for epoch in range(100):
    optimizer.zero_grad()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.float())  # Assurez-vous que les données sont en float
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
    print(f'Époque [{epoch+1}/100], Perte: {loss.item():.4f}')

torch.save(model.state_dict(), 'gaussian_predictor_model.pth')
'''