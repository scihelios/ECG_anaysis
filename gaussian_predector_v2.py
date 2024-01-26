
import torch
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
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing Function
def preprocess_data(inputs, targets):
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    inputs_scaled = input_scaler.fit_transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, targets.shape[-1])).reshape(targets.shape)
    
    return inputs_scaled, targets_scaled, input_scaler, target_scaler

# Enhanced Model Architecture
class EnhancedLinearGaussianPredictor(nn.Module):
    def __init__(self):
        super(EnhancedLinearGaussianPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, x):
        return self.network(x)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
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
    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

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


# Define the load_data function
def load_data(base_path):
    all_sequences = []
    all_targets = []
    for folder_name in tqdm(os.listdir(base_path)):
        folder_path = base_path+'/'+folder_name
        # Check if it's a directory
        if os.path.isdir(folder_path):
            print(folder_path)
            for i in range(len(os.listdir(folder_path))-5):
                test1=[]
                test2 = []
                for j in range(5):
                    with open(os.path.join(folder_path, os.listdir(folder_path)[i+j]), 'r') as file:
                        data = json.load(file)
                        test1 = test1 + data["gaussienne"]
                        time = data["temps_prochain_signal"]*1000
                        test1.append(time)
                with open(os.path.join(folder_path, os.listdir(folder_path)[i+5]), 'r') as file:
                    data = json.load(file)
                    test2 = data["gaussienne"]
                    time = data["temps_prochain_signal"]*1000
                    test2.append(time)
                    all_sequences.append(test1)
                    all_targets.append(test2)

    return np.array(all_sequences), np.array(all_targets)



# Assume the dataset is loaded using the load_data function
folder_path = "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ptb-diagnostic-ecg-database-1.0.0/cleaned_data"
train_inputs, train_targets = load_data(folder_path)
val_inputs, val_targets = load_data(folder_path)  # You can use the same function for validation data
model_save_path = "model.pth"
train_inputs, train_targets, _, _ = preprocess_data(train_inputs, train_targets)
val_inputs, val_targets, _, _ = preprocess_data(val_inputs, val_targets)

# Convert to torch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_targets = torch.tensor(val_targets, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Model
model = EnhancedLinearGaussianPredictor().to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    model.eval()  
    current_input = torch.tensor(initial_data, dtype=torch.float32)
    current_input = current_input.unsqueeze(0)  
    prediction = model(current_input)
    decalage = 808
    with torch.no_grad():
        for _ in range(1,steps):
            print(prediction)
            x_data = np.linspace(0, round(prediction[0].tolist()[12]),round(prediction[0].tolist()[12] ))
            plt.plot(range(round(decalage),round(decalage)+round(prediction[0].tolist()[12])), combined_gaussian(x_data,*np.array(prediction[0].tolist()[0:12])), color ='blue')
            decalage += prediction[0].tolist()[12]
            prediction = model(prediction)
            data.extend(prediction.tolist())
    plt.plot(np.linspace(0, 808,808 ),  combined_gaussian(np.linspace(0, 808,808 ),*np.array(initial_points[0:12])), label='Initial Data')
   
    plt.title("ECG Signal Forecasting")
    plt.xlabel("Time Steps")
    plt.ylabel("ECG Signal Value")
    plt.legend()
    plt.show()
    
    return 

initial_points = []
predicted_points = iterative_forecast(initial_points, model, steps=10)