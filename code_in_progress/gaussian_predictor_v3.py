
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler



def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*3.14)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, params[i],params[5+i],params[10+i]) for i in range(0, 5)])

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EnhancedLinearGaussianPredictor(nn.Module):
    def __init__(self):
        super(EnhancedLinearGaussianPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(45, 128), 
             nn.Linear(128, 15) # First linear layer, expanding to 128 units
  # Second linear layer, reducing to 64 units
        )

    def forward(self, x):
        return self.network(x)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training Function
def train_model(model, train_loader, val_inputs, val_targets, criterion, optimizer, epochs=500, patience=50):
    early_stopping = EarlyStopping(patience=patience)
    val_inputs, val_targets = val_inputs, val_targets

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs, targets

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss.item()}')

        early_stopping(val_loss.item())
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the model
    torch.save(model.state_dict(), "enhanced_model.pth")
    print("Model saved.")


def load_data(base_path, sequence_length=3):
    all_sequences = []
    all_targets = []
    # Assuming the folder contains CSV files
    for folder_name in os.listdir(base_path):
        folder_path = base_path+'/'+folder_name
        df = pd.read_csv(folder_path)
        columns_to_extract = [
            'Centre 1', 'Centre 2', 'Centre 3', 'Centre 4', 'Centre 5',
            'Amplitude 1', 'Amplitude 2', 'Amplitude 3', 'Amplitude 4', 'Amplitude 5',
            'Ecart-type 1', 'Ecart-type 2', 'Ecart-type 3', 'Ecart-type 4', 'Ecart-type 5'
        ]
        heartbeats = []
        for index, row in df.iterrows():
            heartbeat = row[columns_to_extract].tolist()
            heartbeats.append(heartbeat)

        for i in range(len(heartbeats) - sequence_length-1):
            sequence = []
            for j in range(sequence_length):
                sequence = sequence + heartbeats[i+j]
            target = heartbeats[i+sequence_length]
            
            all_sequences.append(sequence)
            all_targets.append(target)
    print(len(all_targets))
    return np.array(all_sequences), np.array(all_targets)



# Assume the dataset is loaded using the load_data function
folder_path = "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/cleaned_csv"
train_inputs, train_targets = load_data(folder_path)
val_inputs, val_targets = train_inputs, train_targets  # You can use the same function for validation data
model_save_path = "model.pth"

# Convert to torch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_targets = torch.tensor(val_targets, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Model
model = EnhancedLinearGaussianPredictor()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Train the model
train_model(model, train_loader, val_inputs, val_targets, criterion, optimizer)

'''
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
    print(f'Validation Loss: {avg_loss}')'''

def iterative_forecast(initial_data, model, steps):
    """
    Predict future sequences iteratively using the LinearGaussianPredictor model.

    :param initial_data: List of initial data points, should be a multiple of 13.
    :param model: Trained LinearGaussianPredictor model.
    :param steps: Number of sequences (each of 13 points) to predict.
    :return: List containing the initial data followed by the predicted sequences.
    """
    data = []
    model.eval()  

    current_input = torch.tensor(initial_data, dtype=torch.float32)
    current_input = current_input.unsqueeze(0)  
    with torch.no_grad():
        for _ in range(1,steps):
            prediction = model(current_input)[0].tolist()
            print(current_input)
            print(prediction)
            decalage = _*6.28
            x_data = np.linspace(decalage-3.14,decalage+3.14,600)
            plt.plot(x_data,combined_gaussian(np.linspace(-3.14,3.14,600),*np.array(prediction)) , color ='red')
            current_input = current_input[0].tolist()[15:45]+prediction
            current_input = torch.tensor(current_input, dtype=torch.float32)
            current_input = current_input.unsqueeze(0) 



    plt.title("ECG Signal Forecasting")
    plt.xlabel("Time Steps")
    plt.ylabel("ECG Signal Value")
    plt.show()
    
    return data

initial_points = [[0.7894443684474293, 1.2623969346137176, 1.690607945857974, 2.885186967144671, 0.419431452606341, -0.2152768511471194, 0.8123947634115295, -1.4594050217912893, -0.066635642069522, -0.1231749640483981, 0.0058671642110163, 0.1270404957002178, 0.1547036511603317, 2.6316533326519405e-05, 0.004975407557482],
 [-1.3843242124389257, -1.1751111728171126, -0.1292836482776922, 0.1422120130776742, 0.142112932290905, 0.6303142219543676, 0.610218585333372, 0.0165421899511318, 0.0360805583965498, 0.1262391384478136, 0.1100529268965558, 0.1010155345342313, 0.0010000214972067, 0.0010000517189971, 0.0233374009256142],
[0.9164742236184472, 1.3213471228785507, 1.699192208641951, 2.550627012665021, 0.6032483926560098, -0.2378631508842055, 0.8041064730079216, -1.3904171919490662, -0.137511185565472, -0.1834783950889455, 0.0039114105953624, 0.1061439534591955, 0.1565679352448626, 0.0035486371183422, 0.0212606039541496], 
[-3.141662097168031, -1.5204251842399084, -1.2919498068745872, -0.2880351476542339, -0.2881941235278917, -0.111305092954336, 0.5296959206158987, 0.60032630860753, -0.1341210759031217, -0.1334861182715094, 0.0033896073130365, 0.0839575751040354, 0.092248785937699, 0.0009996973476107, 0.0123893394341042], 
[0.8887515188727114, 1.3780599479077589, 1.8083083646739664, 2.86360644068474, 0.0136478773846485, -0.1508442344350429, 0.9016863131731148, -1.3207615717825507, -0.0245888377974739, 0.1448556637863657, 0.0062815981663158, 0.1164165796720466, 0.1636022248169472, -0.0026825671780732, 0.0123879586175737]]


input_for_model=[i for j in initial_points for i in j]
predicted_points = iterative_forecast(input_for_model, model, steps=10)




