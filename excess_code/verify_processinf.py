import matplotlib.pyplot as plt
import json

import os
from shutil import copy2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
list_of_errors = []
from tqdm import tqdm
destination_path =  "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/outliers"

def plot_gaussian(x_data, params):
    # Colors to cycle through
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_cycle = iter(colors)

    # Plot each Gaussian component with a different color
    for i in range(0, len(params), 3):
        amplitude, mu, sigma = params[i:i+3]
        current_color = next(color_cycle)  # Get next color from the cycle
        component_y = amplitude * np.exp(-((x_data - mu) ** 2) / (2 * sigma ** 2))
        plt.plot(x_data, component_y, color=current_color)
        plt.axvline(x=mu, color=current_color, linestyle='--')

    # Plot the combined Gaussian
    plt.plot(x_data, combined_gaussian(x_data, *params), color='grey', label='Combined Gaussian')

    plt.legend()
    plt.show()

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))/(np.sqrt(2*3.4)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

def error_function(params, x_data, y_data):
    return np.sum((combined_gaussian(x_data, *params) - y_data) ** 2)

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/data with gaussian info")
count=0

# Loop through each 'Person_xx' directory within the source directory.
for signal in tqdm(os.listdir(src_directory)):
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/data with gaussian info/"+signal, "r") as json_file:
            data = json.load(json_file)
            test = np.array(data["signal"])

        test = test -test[0]
        
        params_progression = []
        # Your original signal data
        x_data = np.linspace(0, len(test),len(test) )  # This is just an example. Replace with your x data.
        y_data = test

        error = error_function(np.array(data['gaussienne']),x_data,y_data)
        list_of_errors.append(error)
        if  error>0.1 and error<0.2:
            #print(np.array(data['gaussienne']))
            plt.plot(x_data,y_data)
            plot_gaussian(x_data,np.array(data['gaussienne']))

        if error>100: 
            shutil.move("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ecg/"+signal, destination_path)
            print(f"File moved successfully from {signal} to {destination_path}")
# Create bins in logarithmic scale
bins = np.logspace(np.log10(min(list_of_errors)), np.log10(max(list_of_errors)), num=20)

# Create the histogram
plt.hist(list_of_errors, bins=bins, edgecolor='black')

# Set the scale of x-axis to logarithmic
plt.xscale('log')

# Add titles and labels
plt.title('Histogram of Values by Order of Magnitude')
plt.xlabel('Value (log scale)')
plt.ylabel('Frequency')

# Show the plot
plt.show()

