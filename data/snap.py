"""
Script permettant de découper les signaux en battements cardiaques et de les ranger dans un dossier
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from tqdm import tqdm


data_folder = f'data/1/full_signal'
output_folder = f'data/1/beats'

def cut_beats(signal):
    """
    Fonction permettant de découper un signal en battements cardiaques
    """
    beats = []

    # On récupère les indices des pics du signal
    peaks, _ = sig.find_peaks(signal, distance=70,height=100)

    # On découpe le signal en battements cardiaques entre chaque pic
    for i in range(len(peaks)):
        # Indice du pic précédent
        if i == 0:
            peak_a = 0
        else:
            peak_a = peaks[i-1]
        # Indice du pic central
        peak_b = peaks[i]
        # Indice du pic suivant
        if i == len(peaks)-1:
            peak_c = len(signal)
        else:
            peak_c = peaks[i+1]
        
        indice_start = (peak_a + peak_b) // 2
        indice_end = (peak_b + peak_c) // 2

        beat = signal[indice_start:indice_end]
        beats.append(beat)

    return beats

""" 
On parcourt tous les fichiers du dossier data_folder et on découpe les signaux en battements cardiaques
On les enregistre ensuite dans le dossier output_folder
"""
j = 1
i = 1
for file in tqdm(np.sort(os.listdir(data_folder))):
    if not os.path.exists(f'{output_folder}/{j}'):
        os.mkdir(f'{output_folder}/{j}')

    if file.endswith('.npy'):
        signal = np.load(f'{data_folder}/{file}')
        beats = cut_beats(signal)
        for beat in beats:
            np.save(f'{output_folder}/{j}/{i}.npy', np.array(beat))
            i+=1
    j+=1
    

print(f'Traitement des signaux terminé. {i-1} battements cardiaques ont été extraits.')