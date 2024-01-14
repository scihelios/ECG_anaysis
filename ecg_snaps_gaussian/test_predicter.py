
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))/(np.sqrt(2*3.4)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

def error_function(params, x_data, y_data):
    return np.sum((combined_gaussian(x_data, *params) - y_data) ** 2)

class GaussianPredictor(nn.Module):
    def __init__(self):
        super(GaussianPredictor, self).__init__()
        # Couche d'entrée: 400 neurones (1 pour chaque point du signal)
        self.fc1 = nn.Linear(400, 128)  # Première couche cachée: 128 neurones
        self.fc2 = nn.Linear(128, 64)   # Deuxième couche cachée: 64 neurones
        self.fc3 = nn.Linear(64, 32)    # Troisième couche cachée: 32 neurones
        # Couche de sortie: 15 neurones (3 paramètres * 5 gaussiennes)
        self.fc4 = nn.Linear(32, 15)    

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation ReLU pour la première couche cachée
        x = F.relu(self.fc2(x))  # Activation ReLU pour la deuxième couche cachée
        x = F.relu(self.fc3(x))  # Activation ReLU pour la troisième couche cachée
        x = self.fc4(x)          # Pas d'activation dans la couche de sortie
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

model = GaussianPredictor()
model.load_state_dict(torch.load('gaussian_predictor_model.pth'))
model.eval()  # Mettez le modèle en mode évaluation

with open("signal61.json", 'r') as file:
    data = json.load(file)
    signal = data["signal"]
    x_data = np.linspace(0, len(signal),len(signal) )  # This is just an example. Replace with your x data.
    y_data = signal
# Convertissez le signal en un tenseur PyTorch
signal_tensor = torch.tensor(signal, dtype=torch.float32)


with torch.no_grad():  # Désactive le calcul du gradient pour l'évaluation
    output = model(signal_tensor)
    output=np.array(list(output))
    print(output)
    plt.plot(x_data,combined_gaussian(x_data,*output))
    plt.show()