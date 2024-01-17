
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
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*3.14)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

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

def iterative_forecast(initial_data, model, steps):
    """
    Predict future sequences iteratively using the LinearGaussianPredictor model.

    :param initial_data: List of initial data points, should be a multiple of 13.
    :param model: Trained LinearGaussianPredictor model.
    :param steps: Number of sequences (each of 13 points) to predict.
    :return: List containing the initial data followed by the predicted sequences.
    """
    data = initial_data.copy()
    model.eval()  
    current_input = torch.tensor(data, dtype=torch.float32)
    current_input = current_input.unsqueeze(0)  
    prediction = model(current_input)
    with torch.no_grad():
        for _ in range(1,steps):
            x_data = np.linspace(0, 700,700 )
            print(prediction[0].tolist()[0:12])
            plt.plot(range(_*700,_*700+700), combined_gaussian(x_data,*np.array(prediction[0].tolist()[0:12])), label='Predicted Data' , color ='blue')
            prediction = model(prediction)
            data.extend(prediction.tolist())
    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, 700,700 ),  combined_gaussian(np.linspace(0, 700,700 ),*np.array(initial_points[0:12])), label='Initial Data')

    plt.title("ECG Signal Forecasting")
    plt.xlabel("Time Steps")
    plt.ylabel("ECG Signal Value")
    plt.legend()
    plt.show()
    
    return data

initial_points = [1.8455169423847149, 188.87683819344306, 17.224703923033996, -52.467565243577106, 352.6206057086834, 16.675476037346876, 64.6044557341793, 355.2796924860764, 15.209228658390366, 16.6898156225738, 594.3673431256359, 43.73143947417923, 0.808]
model = LinearGaussianPredictor()
checkpoint = torch.load("model.pth")
predicted_points = iterative_forecast(initial_points, model, steps=7)

