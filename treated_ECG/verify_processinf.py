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

destination_path =  "C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/outliers"

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))/(np.sqrt(2*3.4)*sigma)

# Define the combined model with 5 Gaussian functions
def combined_gaussian(x, *params):
    return sum([gaussian(x, *params[i:i+3]) for i in range(0, len(params), 3)])

def error_function(params, x_data, y_data):
    return np.sum((combined_gaussian(x_data, *params) - y_data) ** 2)

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ecg")
count=0

# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    if signal.endswith('.json'):
        with open("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/treated_ecg/"+signal, "r") as json_file:
            data = json.load(json_file)
            test = np.array(data["signal"])

        
        params_progression = []
        # Your original signal data
        x_data = np.linspace(0, len(test),len(test) )  # This is just an example. Replace with your x data.
        y_data = test

        error = error_function(np.array(data['gaussienne']),x_data,y_data)

        print(signal)
        list_of_errors.append(error)
        if  error>0.5 and error<1:
            print(np.array(data['gaussienne']))
            plt.plot(x_data,y_data)
            plt.plot(x_data,combined_gaussian(x_data,*np.array(data['gaussienne'])))
            plt.show()

        if error>1: 
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

