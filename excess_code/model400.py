import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np

# Define a custom dataset class
class SignalDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                print(filename)
                with open(os.path.join(directory, filename), 'r') as file:
                    json_data = json.load(file)
                    self.data.append((json_data["signal"], json_data["next_point"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, next_point = self.data[idx]
        return torch.tensor(signal, dtype=torch.float).view(-1, 1), torch.tensor(next_point, dtype=torch.float)

# Assuming each of your data points is a sequence of 400 numbers
sequence_length = 400
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 10

# RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load your data
# Replace 'path_to_your_data_folder' with the actual path to your folder containing JSON files
dataset = SignalDataset('C:/Users/ahmed mansour/Desktop/smaller training')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for signals, next_points in train_loader:
        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, next_points.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model_save_path = 'C:/Users/ahmed mansour/Desktop/rnn_model.pth'
torch.save(model, model_save_path)
print(f'Model saved to {model_save_path}')