
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
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*3.14)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

# Define the linear model
# Define the linear model
class LinearGaussianPredictor(nn.Module):
    def __init__(self):
        super(LinearGaussianPredictor, self).__init__()
        # Linear layer with 16 inputs and 16 outputs
        self.linear = nn.Linear(80, 16)

    def forward(self, x):
        # Pass the input through the linear layer
        x = self.linear(x)
        return x

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

    return np.array(all_sequences, dtype=np.float32), np.array(all_targets, dtype=np.float32)

# Instantiate the model
model = LinearGaussianPredictor()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Assume the dataset is loaded using the load_data function
folder_path = "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ptb-diagnostic-ecg-database-1.0.0/cleaned_data"
train_inputs, train_targets = load_data(folder_path)
val_inputs, val_targets = load_data(folder_path)  # You can use the same function for validation data
model_save_path = "model.pth"
# Training loop
epochs = 50
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
    
    return data

initial_points = [1.8455169423847149, 188.87683819344306, 17.224703923033996, -52.467565243577106, 352.6206057086834, 16.675476037346876, 64.6044557341793, 355.2796924860764, 15.209228658390366, 16.6898156225738, 594.3673431256359, 43.73143947417923, 808]
predicted_points = iterative_forecast(initial_points, model, steps=50)