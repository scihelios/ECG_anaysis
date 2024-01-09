import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(400, 400)  # Input layer of size 400, hidden layer of size 50
        # Activation function
        self.relu = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(400, 1)  # Hidden layer of size 50, output layer of size 10

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU activation function
        x = self.fc2(x)
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
                params = data["next_point"]
                all_inputs.append(torch.tensor(signals))
                all_targets.append(torch.tensor(params))

    
    all_inputs = torch.stack(all_inputs)
    all_targets = torch.stack(all_targets)
    return all_inputs, all_targets

# Exemple : remplacer 'your_folder_path' par le chemin de votre dossier
inputs, targets = load_data('C:/Users/ahmed mansour/Desktop/smaller training')

from sklearn.model_selection import train_test_split

# Assuming 'all_inputs' and 'all_targets' are your complete dataset
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42)  # 20% data for validation

from torch.utils.data import DataLoader, TensorDataset

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

from torch.optim.lr_scheduler import StepLR

# Assuming GaussianPredictor, dataloader, and validation_dataloader are already defined

model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)  # Adjust the step size and gamma as needed

num_epochs = 10000
best_val_loss = float('inf')
patience, trials = 1000, 0  # Early stopping parameters


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
