
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Convertissez le signal en un tenseur PyTorch
signal_tensor = torch.tensor(signal, dtype=torch.float32)


with torch.no_grad():  # Désactive le calcul du gradient pour l'évaluation
    output = model(signal_tensor)
    print(output)