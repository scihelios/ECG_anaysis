"""
Script pour extraire les signaux de la base de données et les mettre dans un fichier .csv
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bdd_folder = f'data/1/bdd'
output_folder = f'data/1/full_signal'


"""
1. Extraction des signaux
On parcourt la base de données et on récupère tous les signaux que l'on met dans un nouveau dossier sous forme de fichier .csv
"""


i = 1
for person in np.sort(os.listdir(bdd_folder)):
    for file in np.sort(os.listdir(f'{bdd_folder}/{person}')):
        if file.endswith('.dat'):
            df = np.fromfile(f'{bdd_folder}/{person}/{file}', dtype=np.int16)
            np.save(f'{output_folder}/{i}', df)
            i+=1
