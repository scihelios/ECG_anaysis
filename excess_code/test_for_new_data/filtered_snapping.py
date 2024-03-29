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


def extract_cycles(signal):
    # Détecter les pics R. Vous pourriez avoir besoin d'ajuster la distance et la hauteur selon la qualité de votre signal.
    peaks, _ = find_peaks(signal, distance=150, height=0.55)
    
    # Découper le signal en cycles
    cycles = [signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
    
    return cycles, peaks

def extract__deriv_cycles(signal):
    # Détecter les pics R. Vous pourriez avoir besoin d'ajuster la distance et la hauteur selon la qualité de votre signal.
    peaks, _ = find_peaks(signal, distance=150, height=0.5)
    
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
    signal = data[1::12]

    return  signal/max(signal)


count=0
count_d = 0

signal = load_signals('C:/Users/ahmed mansour/Desktop/scolarite X/2A/Psc/ECG_anaysis/excess_code/s0543_re.dat')
cycles, peaks = extract_cycles(signal)
print(signal)
from pyts.decomposition import SingularSpectrumAnalysis
Y = signal.reshape(1,-1)
ssa = SingularSpectrumAnalysis(window_size=25, groups='auto')
Y_ssa = ssa.fit_transform(Y)

signal = Y_ssa[0,0]

deriv = np.array([signal[i] - signal[i-1] for i in range(1,len(signal))])
coef_of_max = max(deriv)
deriv= deriv/coef_of_max
cycles, peaks = extract__deriv_cycles(deriv)
deriv = deriv *coef_of_max
last_peak =1 

interval = round(len(signal)/(2*len(peaks)))


for i in peaks[1:len(peaks)-2]:
    last_peak+=1
    start = signal[i-interval]
    ecg=[start]
    local_deriv =deriv[i-interval:i+interval]
    for j in local_deriv:
        ecg.append(j+ecg[-1])
    ecg=np.array(ecg)
    with open("signal_p_1_N"+str(count_d)+".json", "w") as json_file:
        json.dump({"signal":list(ecg) , "gaussienne":[],"temps_prochain_signal":[(peaks[last_peak]-i)/1000]}, json_file)
    count_d +=1


