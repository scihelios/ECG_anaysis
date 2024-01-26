
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

def plot_gaussian(x_data,param):
    plt.plot(x_data,combined_gaussian(x_data,*param))
    return

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*3.14)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

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

initial_points = [[27.73777368349089, 292.8181952224694, 149.85349954322572, -5.661706668628136, 335.5080433230316, 27.46301061589346, 94.04513550710266, 425.40265776302135, 18.217623633879573, -87.16209119022356, 426.4745616390194, 20.003121695059924, 19.127249527067647, 681.3654794066061, 44.76426172259344,0.801],
[2.040217483314305, 256.03565983656483, 14.559489626611999, -5.7386374408086045, 345.5074116840866, 41.001689901692764, 94.08627976642777, 426.9564267941113, 18.6448514316938, -87.02233347743677, 428.18744586404347, 20.655807347812093, 10.669499773256197, 684.2260537986747, 34.07445568900331,0.816],
[6.459944309283847, 241.21012534316074, 34.51699120208273, -3.655232734724498, 348.2161877912265, 39.43704036190897, 96.49650759636111, 428.36839763264936, 16.07037940679518, -84.92857272102015, 429.55231581722217, 16.691258148666392, 14.98915631198527, 683.866904066573, 41.492589251706924,0.818],
[7.7125519137270375, 250.06603783382795, 33.44120271918599, -22.794387854894673, 269.82288699751047, 126.5276219924318, 94.66100209857777, 426.5872478879902, 16.953385971259063, -84.99270932818627, 427.6344191047927, 18.204151785342873, 16.050620114797763, 676.8739886410069, 38.82430541990137,0.815],
[1.6651171572045196, 252.03571220845842, 12.958739637321052, -2.1313926481424734, 344.2264601948587, 42.261688386427764, 95.83339698813073, 426.9085088201338, 17.41759272522989, -85.50200480216003, 428.466074316189, 18.484036488130776, 21.75608656960837, 679.4311201603186, 45.397523847690486,0.819]]
decalage=0
for i in initial_points:
    num_points=round(i[15]*1000)
    plt.plot(np.linspace(decalage,decalage+num_points,num_points),combined_gaussian(np.linspace(0,num_points,num_points),*np.array(i[0:15])),color ='blue')
    decalage+=num_points
plt.show()
model =EnhancedLinearGaussianPredictor()
checkpoint = torch.load("enhaced_model.pth")
predicted_points = iterative_forecast(initial_points, model, steps=7)

