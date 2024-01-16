

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

# Define the linear model
class LinearGaussianPredictor(nn.Module):
    def __init__(self):
        super(LinearGaussianPredictor, self).__init__()
        # Linear layer with 16 inputs and 16 outputs
        self.linear = nn.Linear(13, 13)

    def forward(self, x):
        # Pass the input through the linear layer
        x = self.linear(x)
        return x

# Define the load_data function
def load_data(folder_path):
    all_sequences = []
    all_targets = []
    for i in range(len(os.listdir(folder_path))):
        if os.listdir(folder_path)[i].startswith('signal') and os.listdir(folder_path)[i].endswith('.json') and i< 100:
            with open(os.path.join(folder_path, os.listdir(folder_path)[i]), 'r') as file:
                data = json.load(file)
                test1 = data["gaussienne"]
                time = data["temps_prochain_signal"]
                test1.append(time)
            with open(os.path.join(folder_path, os.listdir(folder_path)[i+1]), 'r') as file:
                data = json.load(file)
                test2 = data["gaussienne"]
                time = data["temps_prochain_signal"]
                test2.append(time)
                all_sequences.append(test1)
                all_targets.append(test2)

    return np.array(all_sequences), np.array(all_targets)

# Instantiate the model
model = LinearGaussianPredictor()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Assume the dataset is loaded using the load_data function
folder_path = "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/data with gaussian info"
train_inputs, train_targets = load_data(folder_path)
val_inputs, val_targets = load_data(folder_path)  # You can use the same function for validation data
model_save_path = "model.pth"
# Training loop
epochs = 30
for epoch in range(epochs):
    model.train() # Set the model to training mode
    for i in range(len(train_inputs)):
        # Convert data to tensors
        inputs = torch.tensor(train_inputs[i], dtype=torch.float32)
        targets = torch.tensor(train_targets[i], dtype=torch.float32)

        # Forward pass: compute predicted output by passing inputs to the model
        predictions = model(inputs)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Perform backpropagation
        optimizer.step()       # Update the parameters
        
        # Optionally print the loss every 'print_every' steps
        print_every = 10  # Adjust as needed
        if (i + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_inputs)}], Loss: {loss.item()}')
    if epoch == epochs-1:
        torch.save(model.state_dict(),model_save_path )
        print(f'Saved model at epoch {epoch+1}')
# Evaluation
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    total_loss = 0
    for i in range(len(val_inputs)):
        inputs = torch.tensor(val_inputs[i], dtype=torch.float32)
        targets = torch.tensor(val_targets[i], dtype=torch.float32)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        total_loss += loss.item()
    avg_loss = total_loss / len(val_inputs)
    print(f'Validation Loss: {avg_loss}')