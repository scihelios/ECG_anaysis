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
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/test_for_new_data")
count=0

y_data = np.zeros(701)
for signal in os.listdir(src_directory):
    print(os.listdir(src_directory))
    plt.show()
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/test_for_new_data/"+signal, "r") as json_file:
            data = json.load(json_file)
            test = np.array(data["signal"])
            time = data["temps_prochain_signal"][0]
        x_data = np.linspace(0, len(test),len(test) )
        y_data+=test
params_progression = []
y_data = y_data/151
y_data = y_data + -y_data[0]
peak = 360
initial_guess = [ 18.31325284, peak *0.5,  13.7655355 ,
                    -2.20945481, peak *0.85 , 10.09754921, 
                    29.42391721, peak ,  62.27029317,  
                    10.10585412, peak*1.5 ,  25.89649657 ,   ]
# Loop through each 'Person_xx' directory within the source directory.

# Fit the data
res = minimize(error_function,initial_guess,args=(x_data, y_data),tol=1e-7,options={'maxiter': 100000},callback=callback)

initial_guess =list(res.x)

for signal in os.listdir(src_directory):
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/test_for_new_data/"+signal, "r") as json_file:
            data = json.load(json_file)
            test = np.array(data["signal"])
            time = data["temps_prochain_signal"][0]
        params_progression = []
        longeur_signal = len(test)//2
        # Your original signal data
        x_data = np.linspace(0, len(test),len(test) )  # This is just an example. Replace with your x data.
        y_data = test - test[0]

        # Fit the data
        res = minimize(error_function,initial_guess,args=(x_data, y_data),tol=1e-7,options={'maxiter': 100000},callback=callback)


        with open(signal, "w") as json_file:
            json.dump({"signal":list(test) , "gaussienne":list(res.x), "temps_prochain_signal":time} , json_file)
        count+=1
        print(count)