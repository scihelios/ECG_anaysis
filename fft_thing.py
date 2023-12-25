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

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))(np.sqrt(2*3.14)*sigma)

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

for signal in os.listdir(src_directory):

    if signal.endswith('.json') :
        with open('treated_ECG/'+signal, "r") as json_file:
            data = json.load(json_file)
        new_data = data['signal'].copy()
        
        # Replace 'your_signal' with your 400 data points
        your_signal = new_data # Example signal, replace this with your data

        # Perform FFT
        fft_result = np.fft.fft(your_signal)
        fft_result *= np.abs(fft_result)>0.1
        print(fft_result.shape)

        # Perform Inverse FFT to reconstruct the signal
        reconstructed_signal = np.fft.ifft(fft_result)

        # Since the original signal is real, the imaginary part of the reconstructed signal should be near zero
        # However, due to numerical computation, it's possible to have a tiny imaginary part
        # We'll take just the real part for plotting
        reconstructed_signal = reconstructed_signal.real

        # Plotting
        plt.figure(figsize=(18, 6))

        # Plot the original signal
        plt.subplot(1, 3, 1)
        plt.scatter(list(range(len(your_signal))),your_signal, linewidths=1e-4)
        plt.title("Original Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # Plot the magnitude of FFT
        plt.subplot(1, 3, 2)
        plt.plot(np.fft.fftfreq(len(your_signal)), np.abs(fft_result))
        plt.title("Magnitude of FFT")
        plt.xlabel("Frequency")
        plt.yscale("log")
        plt.ylabel("Magnitude")

        # Plot the reconstructed signal
        plt.subplot(1, 3, 3)
        plt.plot(reconstructed_signal)
        plt.title("Reconstructed Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        count+=1
        print(count)
