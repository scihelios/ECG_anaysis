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



# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/test_for_new_data")
count=0
y_data = np.zeros(701)
# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    print(signal)
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/test_for_new_data/"+signal, "r") as json_file:
            print('hey')
            data = json.load(json_file)
            test = np.array(data["signal"])
            time = data["temps_prochain_signal"][0]
        x_data = np.linspace(0, len(test),len(test) )
        y_data+=test

y_data = y_data/151
plt.plot(x_data,y_data)
plt.show()