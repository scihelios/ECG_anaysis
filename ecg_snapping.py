import os
from shutil import copy2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Charger le signal
def load_signals(filename):
    data = np.fromfile(filename, dtype=np.int16)
    raw_signal = data[::2]
    filtered_signal = data[1::2]
    return raw_signal, filtered_signal

def extract_cycles(signal):
    # Détecter les pics R. Vous pourriez avoir besoin d'ajuster la distance et la hauteur selon la qualité de votre signal.
    peaks, _ = find_peaks(signal, distance=50, height=70)
    
    # Découper le signal en cycles
    cycles = [signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
    
    return cycles, peaks

import json


# Load the signals
def load_signals(filename):
    # Since the data is stored as 16-bit integers
    data = np.fromfile(filename, dtype=np.int16)

    # Split the data into the two signals (raw and filtered)
    # Assuming interleave format
    raw_signal = data[::2]
    filtered_signal = data[1::2]

    return raw_signal, filtered_signal

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data")
count=0

# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    # Charger le signal brut
    raw_signal, _ = load_signals("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data/"+signal)
    cycles, peaks = extract_cycles(raw_signal)
    for i in cycles:
        ecg = i
        test = np.concatenate((ecg[len(ecg)//2:len(ecg+1)],ecg[0:len(ecg)//2] ))
        test = np.array(test)/100

        with open("signal"+str(count)+".json", "w") as json_file:
            json.dump({"signal":list(test) , "gaussienne":[]}, json_file)
        count+=1
        print(count)


#raw_signal, filtered_signal = load_signals()