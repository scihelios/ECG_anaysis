import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

df = pd.read_csv('data/1/parametres.csv')

# Copie du DataFrame pour nettoyer les données sans affecter le DataFrame original
df_clean = df.copy()

# Supprimer les valeurs aberrantes
for i, col in enumerate(df.columns):
    mean = df[col].mean()
    std = df[col].std()
    if i <= 4:
        nb_ecarts_types = 1
    if i >= 5 and i <= 9:
        nb_ecarts_types = 5
    else:
        nb_ecarts_types = 2
    upper_bound = mean + nb_ecarts_types * std
    lower_bound = mean - nb_ecarts_types * std
    df_clean[col] = df[col].clip(lower_bound, upper_bound)

# Créer un dataframe avec chaque colonne comme paramètre, premiere ligne la moyenne, et deuxieme ligne l'ecart type
df_mean_std = {col:[] for col in df.columns}
df_mean_std = pd.DataFrame(df_mean_std)

# Créer une figure et un ensemble de sous-graphiques avec 3 lignes et 5 colonnes
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))

# Aplatir le tableau d'axes pour faciliter l'itération
axes = axes.flatten()

for i, col in enumerate(df_clean.columns):
    # Tracer l'histogramme pour chaque colonne
    data = df_clean[col].dropna()  # Supprimer les valeurs NaN
    ax = axes[i]
    data.hist(ax=ax, bins=20, edgecolor='black', density=False)
    
    # Calculer la moyenne et l'écart-type
    mu, std = norm.fit(data)

    # Ajouter la moyenne et l'écart type dans le dataframe
    df_mean_std.loc[0, col] = mu
    df_mean_std.loc[1, col] = std
    
    # Tracer la courbe gaussienne
    n, bins, patches = ax.hist(data, bins=20, edgecolor='black')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    p_scaled = p * len(data) * np.diff(bins[:2])  # Multiplier par le nombre total d'échantillons et la largeur des bins
    ax.plot(x, p_scaled, 'k', linewidth=2)

    # Ajouter un titre et les légendes
    ax.set_title(col)

# Ajuster l'espacement entre les sous-graphiques
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Afficher le graphique
plt.show()
plt.close()

# Calcul des coefficient de correlation
tab_corr = df_clean.corr()

# Affichage des coefficient de correlation

sns.heatmap(tab_corr, annot=True, cmap='RdBu_r', center=0)
plt.show()
plt.close()

# Calcul du tabeau de covariance

tab_cov = tab_corr.to_numpy()
np_mean_std = df_mean_std.to_numpy()
n = tab_cov.shape[0]
for i in range(n):
    for j in range(n):
        tab_cov[i, j] = tab_cov[i, j] * np_mean_std[1, i] * np_mean_std[1, j]

tab_cov = pd.DataFrame(tab_cov)
tab_cov.to_csv('data/1/cov.csv')

tab_corr.to_csv('data/1/corr.csv')

# Affichage des covariances

sns.heatmap(tab_cov, annot=True, cmap='RdBu_r', center=0)
plt.show()

def creat_x0():

    x0 = np.zeros((1, 18))
    x0[0, 1] = -np.pi
    x0[0, 2] = 1

    for i in range(15):
        x0[0, i+3] = np_mean_std[0, i]

    return x0

print(creat_x0())