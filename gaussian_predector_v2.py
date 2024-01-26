
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



class EnhancedLinearGaussianPredictor(nn.Module):
    def __init__(self):
        super(EnhancedLinearGaussianPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(80, 128),
              nn.Linear(128, 16),  # First linear layer, expanding to 128 units
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
def train_model(model, train_loader, val_inputs, val_targets, criterion, optimizer, epochs=1500, patience=100):
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


# Define the load_data function
def load_data(base_path):
    s=0
    all_sequences = []
    all_targets = []
    for folder_name in os.listdir(base_path):
        folder_path = base_path+'/'+folder_name
        
        # Check if it's a directory
        if os.path.isdir(folder_path) and s<10:
            s+=1
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
val_inputs, val_targets = train_inputs, train_targets  # You can use the same function for validation data
model_save_path = "model.pth"

# Convert to torch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_targets = torch.tensor(val_targets, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model
model = EnhancedLinearGaussianPredictor()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

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

def iterative_forecast(initial_data, model,decalage, steps):
    """
    Predict future sequences iteratively using the LinearGaussianPredictor model.

    :param initial_data: List of initial data points, should be a multiple of 13.
    :param model: Trained LinearGaussianPredictor model.
    :param steps: Number of sequences (each of 13 points) to predict.
    :return: List containing the initial data followed by the predicted sequences.
    """
    data = []
    model.eval()  
    print(decalage)
    current_input = torch.tensor(initial_data, dtype=torch.float32)
    current_input = current_input.unsqueeze(0)  
    with torch.no_grad():
        for _ in range(1,steps):
            prediction = model(current_input)[0].tolist()
            print(current_input)
            print(prediction)
            num_points = round(prediction[15])
            x_data = np.linspace(decalage,decalage+num_points,num_points)
            plt.plot(x_data, combined_gaussian(np.linspace(0,num_points,num_points),*np.array(prediction[0:15])) , color ='red')
            decalage += num_points
            current_input = current_input[0].tolist()[16:80]+prediction
            current_input = torch.tensor(current_input, dtype=torch.float32)
            current_input = current_input.unsqueeze(0) 



    plt.title("ECG Signal Forecasting")
    plt.xlabel("Time Steps")
    plt.ylabel("ECG Signal Value")
    plt.show()
    
    return data

initial_points = [[27.73777368349089, 292.8181952224694, 149.85349954322572, -5.661706668628136, 335.5080433230316, 27.46301061589346, 94.04513550710266, 425.40265776302135, 18.217623633879573, -87.16209119022356, 426.4745616390194, 20.003121695059924, 19.127249527067647, 681.3654794066061, 44.76426172259344,801],
[2.040217483314305, 256.03565983656483, 14.559489626611999, -5.7386374408086045, 345.5074116840866, 41.001689901692764, 94.08627976642777, 426.9564267941113, 18.6448514316938, -87.02233347743677, 428.18744586404347, 20.655807347812093, 10.669499773256197, 684.2260537986747, 34.07445568900331,816],
[6.459944309283847, 241.21012534316074, 34.51699120208273, -3.655232734724498, 348.2161877912265, 39.43704036190897, 96.49650759636111, 428.36839763264936, 16.07037940679518, -84.92857272102015, 429.55231581722217, 16.691258148666392, 14.98915631198527, 683.866904066573, 41.492589251706924,818],[7.7125519137270375, 250.06603783382795, 33.44120271918599, -22.794387854894673, 269.82288699751047, 126.5276219924318, 94.66100209857777, 426.5872478879902, 16.953385971259063, -84.99270932818627, 427.6344191047927, 18.204151785342873, 16.050620114797763, 676.8739886410069, 38.82430541990137,815],
[1.6651171572045196, 252.03571220845842, 12.958739637321052, -2.1313926481424734, 344.2264601948587, 42.261688386427764, 95.83339698813073, 426.9085088201338, 17.41759272522989, -85.50200480216003, 428.466074316189, 18.484036488130776, 21.75608656960837, 679.4311201603186, 45.397523847690486,819]]
decalage=0
for i in initial_points:
    num_points=i[15]
    plt.plot(np.linspace(decalage,decalage+num_points,num_points),combined_gaussian(np.linspace(0,num_points,num_points),*np.array(i[0:15])),color ='blue')
    decalage+=num_points

input_for_model=[i for j in initial_points for i in j]
predicted_points = iterative_forecast(input_for_model, model,decalage, steps=10)

'''[7.7125519137270375, 250.06603783382795, 33.44120271918599, -22.794387854894673, 269.82288699751047, 126.5276219924318, 94.66100209857777, 426.5872478879902, 16.953385971259063, -84.99270932818627, 427.6344191047927, 18.204151785342873, 16.050620114797763, 676.8739886410069, 38.82430541990137,815],
[1.6651171572045196, 252.03571220845842, 12.958739637321052, -2.1313926481424734, 344.2264601948587, 42.261688386427764, 95.83339698813073, 426.9085088201338, 17.41759272522989, -85.50200480216003, 428.466074316189, 18.484036488130776, 21.75608656960837, 679.4311201603186, 45.397523847690486,819]'''