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

def extract_cycles(signal):
    # Détecter les pics R. Vous pourriez avoir besoin d'ajuster la distance et la hauteur selon la qualité de votre signal.
    peaks, _ = find_peaks(signal, distance=100, height=0.55)
    
    # Découper le signal en cycles
    cycles = [signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
    
    return cycles, peaks

def extract__deriv_cycles(signal):
    # Détecter les pics R. Vous pourriez avoir besoin d'ajuster la distance et la hauteur selon la qualité de votre signal.
    peaks, _ = find_peaks(signal, distance=100, height=0.5)
    
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

    return raw_signal/max(raw_signal), filtered_signal/max(filtered_signal)

# Define the source directory where the 'Person_xx' folders are located.
src_directory = Path("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data")
count=0
count_d = 0
# Loop through each 'Person_xx' directory within the source directory.
for signal in os.listdir(src_directory):
    # Charger le signal brut
    _, filtered_signal = load_signals("C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/collected data/"+signal)
    '''
   # Afficher le signal et les pics R détectés
    plt.plot(filtered_signal, label='Signal')
    plt.plot(peaks, filtered_signal[peaks], "xr", label='Pics R')
    plt.legend()
    plt.show()

    for i in cycles:
        ecg = i
        test = np.concatenate((ecg[len(ecg)//2:len(ecg+1)],ecg[0:len(ecg)//2] ))
        with open("signal"+str(count)+".json", "w") as json_file:
            json.dump({"signal":list(test) , "gaussienne":[]}, json_file)
        count+=1
        print(count)'''
    from pyts.decomposition import SingularSpectrumAnalysis
    Y = filtered_signal.reshape(1,-1)
    ssa = SingularSpectrumAnalysis(window_size=50, groups='auto')
    Y_ssa = ssa.fit_transform(Y)

    filtered_signal = Y_ssa[0,0]

    for i in range (len(filtered_signal)-500):
        with open("signal"+str(count_d)+".json", "w") as json_file:
            json.dump({"signal":list(filtered_signal[i:i+400]) , "next_point":filtered_signal[i+400:i+410]}, json_file)
        count_d +=1
        print(count_d, end=" ")