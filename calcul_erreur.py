# FICHIER POUR LE CALCUL DES PERFORMANCES



import kalman as kal
import numpy as np
import matplotlib.pyplot as plt
import parametres as param
import linsub as lin
import extremum as ext
import pandas as pd

label = ['Erreur L^2', 'Erreur L^1', 'Erreur L^inf', 'Erreur a1', 'Erreur a2', 'Erreur a3', 'Erreur a4', 'Erreur a5', 'Erreur m1', 'Erreur m2', 'Erreur m3', 'Erreur m4', 'Erreur m5', 'Erreur s1', 'Erreur s2', 'Erreur s3', 'Erreur s4', 'Erreur s5']
erreur_kalman = pd.DataFrame(columns = label)
erreur_gradient = pd.DataFrame(columns = label)

n = 800
noise_level = 0.02

for j in range(300):
    print(j)
    param_target = kal.generation_parametres()
    signal = param_target.signal_gaussiennes(n)
    signal += noise_level*np.random.randn(n)

    param_gradient,_ = ext.gradient_descent_calibre(signal)
    param_kalman = kal.kalman_interface(signal, 3)
    
    liste_erreur = []
    liste_erreur.append(np.linalg.norm(signal - param_gradient.signal_gaussiennes(n)))
    liste_erreur.append(np.mean(np.abs(signal - param_gradient.signal_gaussiennes(n))))
    liste_erreur.append(np.max(np.abs(signal - param_gradient.signal_gaussiennes(n))))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.amplitudes[i] - param_gradient.amplitudes[i]))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.centres[i] - param_gradient.centres[i]))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.ecarts_types[i] - param_gradient.ecarts_types[i]))
    erreur_gradient.loc[j] = liste_erreur

    liste_erreur = []
    liste_erreur.append(np.linalg.norm(signal - param_kalman.signal_gaussiennes(n)))
    liste_erreur.append(np.mean(np.abs(signal - param_kalman.signal_gaussiennes(n))))
    liste_erreur.append(np.max(np.abs(signal - param_kalman.signal_gaussiennes(n))))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.amplitudes[i] - param_kalman.amplitudes[i]))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.centres[i] - param_kalman.centres[i]))
    for i in range(5):
        liste_erreur.append(np.abs(param_target.ecarts_types[i] - param_kalman.ecarts_types[i]))
    erreur_kalman.loc[j] = liste_erreur



liste_erreur_kalman = erreur_kalman.mean().tolist()
liste_erreur_gradient = erreur_gradient.mean().tolist()

liste_erreur_kalman_std = erreur_kalman.std().tolist()
liste_erreur_gradient_std = erreur_gradient.std().tolist()

# Create a latex tab for the mean and std errors


recap = pd.DataFrame()
recap['Erreur'] = label
recap['Moyenne Kalman'] = liste_erreur_kalman
recap['Moyenne Gradient'] = liste_erreur_gradient
recap['Std Kalman'] = liste_erreur_kalman_std
recap['Std Gradient'] = liste_erreur_gradient_std

print(recap.to_latex(index=False))