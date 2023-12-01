import os
from shutil import copy2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json


from scipy.optimize import minimize
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
error_progression = []
# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*3.14)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

def error_function(params, x_data, y_data):
    return np.sum((combined_gaussian(x_data, *params) - y_data) ** 4)

def callback(params):
    params_progression.append(params.copy())
    error = error_function(params, x_data, y_data)
    error_progression.append(error)


# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/ecg_snaps")
count=0

# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/ecg_snaps/"+signal, "r") as json_file:
            data = json.load(json_file)
            test = np.array(data["signal"])
        params_progression = []
        # Your original signal data
        x_data = np.linspace(0, len(test),len(test) )  # This is just an example. Replace with your x data.
        y_data = test
        initial_guess = []
        for _ in range(5):
            initial_guess.extend([5, _ *len(test)/6 + len(test)/7, 6])

        # Fit the data
        res = minimize(error_function, initial_guess, args=(x_data, y_data), callback=callback)

        with open(signal, "w") as json_file:
            json.dump({"signal":list(test) , "gaussienne":list(res.x)}, json_file)
        count+=1
        print(count)