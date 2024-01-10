import os
from shutil import copy2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import random
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Charger le signal
def load_signals(filename):
    data = np.fromfile(filename, dtype=np.int16)
    raw_signal = data[::2]
    filtered_signal = data[1::2]
    return raw_signal, filtered_signal

import json



# Load the signals
def load_signals(filename):
    # Since the data is stored as 16-bit integers
    data = np.fromfile(filename, dtype=np.int16)

    # Split the data into the two signals (raw and filtered)
    # Assuming interleave format
    raw_signal = data[::2]
    filtered_signal = data[1::2]

    return raw_signal/max(raw_signal), filtered_signal/max(filtered_signal)

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data")
count=0
count_d = 0
# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    # Charger le signal brut
    _, filtered_signal = load_signals("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data/"+signal)
    with open("signal"+str(count)+".json", "w") as json_file:
        json.dump({"signal":list(filtered_signal)}, json_file)
    count+=1
    print(count)