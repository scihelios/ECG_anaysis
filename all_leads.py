
# Importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Importation des données
def get_leads():
    """
    Fonction pour obtenir les leads de la base de données 3
    """
    # Création d'un DataFrame vide
    df = pd.DataFrame()
    
    # Parcours des dossiers de la base de données 3
    for folder in os.listdir('data/3'):
        # Parcours des fichiers de chaque dossier
        for file in os.listdir('data/3/' + folder):
            # Si le fichier est un lead
            if 'lead' in file:
                # Création d'un DataFrame temporaire
                df_temp = pd.read_csv('data/3/' + folder + '/' + file)
                # Ajout du lead au DataFrame principal
                df = pd.concat([df, df_temp], axis=1)
    
    return df

# Création d'un DataFrame avec les leads
df_leads = get_leads()

# Affichage des leads
plt.figure(figsize=(20, 10))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.plot(df_leads.iloc[:, i])
    plt.title('Lead ' + str(i + 1))
    plt.xlabel('Temps (ms)')
    plt.ylabel('Amplitude')
plt.show()