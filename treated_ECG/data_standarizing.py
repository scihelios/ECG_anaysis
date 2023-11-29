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
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

import numpy as np
from scipy import interpolate

def standardize_data(data, target_length=400):
    signal = np.array(data["signal"])
    current_length = len(signal)

    current_indices = np.linspace(0, 1, current_length)

    target_indices = np.linspace(0, 1, target_length)

    coeff_of_change = target_length/current_length
    param = data["gaussienne"]
    for i in range(0,len(param),3):
        param[i+1] = param[i+1]*coeff_of_change
        param[i+2] = param[i+2]*coeff_of_change

    interp_func = interpolate.interp1d(current_indices, signal, kind='linear')

    standardized_signal = interp_func(target_indices)

    return (standardized_signal , param.copy())

src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ECG")
count=0
stop=1

for signal in os.listdir(src_directory):
    if signal.endswith('.json') and stop<30000:
        with open(signal, "r") as json_file:
            data = json.load(json_file)
        print(data)
        new_data = standardize_data(data.copy())
        print(new_data)
        with open(signal, "w") as json_file:
            json.dump({"signal":list(new_data[0]) , "gaussienne":list(new_data[1])}, json_file)
        count+=1
        print(count)
    stop += 1